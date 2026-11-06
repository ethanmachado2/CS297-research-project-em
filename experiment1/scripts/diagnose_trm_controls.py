"""Small real-training-subset controls; no validation/test fitting or scoring.

These controls deliberately change one part of the diagnostic model. They never
replace the production configuration or checkpoint. Each gets the same number
of AdamW updates on the same two people / eight images.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.trm_data import FaceSeverityDataset, file_sha256, partition_frame
from src.trm_model import VisionTRM
from src.trm_training import atomic_json, load_checkpoint, seed_everything


class EncoderClassifier(nn.Module):
    def __init__(self, original, moments=False):
        super().__init__()
        self.encoder = copy.deepcopy(original.encoder)
        self.moments = moments
        if moments:
            old = self.encoder.projection
            new = nn.Linear(old.in_features * 2, old.out_features)
            with torch.no_grad():
                new.weight[:, :old.in_features].copy_(old.weight)
                new.weight[:, old.in_features:].zero_()
                new.bias.copy_(old.bias)
            self.encoder.projection = new
        self.classifier = copy.deepcopy(original.classifier)

    def forward(self, patches):
        if not self.moments:
            vectors = self.encoder(patches)
        else:
            batch, count, channels, height, width = patches.shape
            feature = self.encoder.conv[:-1](patches.reshape(batch * count, channels, height, width))
            moments = torch.cat((feature.mean((2, 3)), feature.std((2, 3), correction=0)), -1)
            vectors = self.encoder.projection(moments).reshape(batch, count, -1)
        return self.classifier(vectors.mean(1))


def measure(model, patches, targets, encoded=None):
    with torch.inference_mode():
        if encoded is None:
            logits = model(patches)
        else:
            state = model.initial_state(len(targets))
            for _ in range(model.config["refinement_steps"]):
                state, logits = model.refine_once(encoded, state)
        return {"loss": nn.functional.cross_entropy(logits, targets).item(),
                "accuracy": (logits.argmax(-1) == targets).double().mean().item(),
                "probability_image_std_rms": logits.softmax(-1).std(0, correction=0).square().mean().sqrt().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/trm_supervised_runs/20261024_implementation_validation/checkpoints/best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--variants", nargs="+", default=["original", "shallow", "normalized_frozen_tokens", "cnn_mean", "cnn_mean_std"])
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("updates must be positive")
    allowed = {"original", "shallow", "h1", "l1", "k1", "raw_frozen_tokens", "normalized_frozen_tokens", "cnn_mean", "cnn_mean_std"}
    if set(args.variants) - allowed:
        parser.error("Unknown variant")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    _, checkpoint = load_checkpoint(args.checkpoint, "cpu")
    config = checkpoint["config"]
    frame = partition_frame(checkpoint["split"], "train", ROOT)
    people = frame.subject_id.drop_duplicates().iloc[:2].tolist()
    selected = frame.loc[frame.subject_id.isin(people)]
    for row in selected.itertuples():
        if file_sha256(row.file_path) != row.sha256:
            raise ValueError(f"Image changed: {row.file_path}")
    dataset = FaceSeverityDataset(selected, config["preprocessing"])
    pairs = [dataset[i] for i in range(len(dataset))]
    patches, targets = torch.stack([p[0] for p in pairs]), torch.tensor([p[1] for p in pairs])
    seed_everything(config["seed"])
    original = VisionTRM(config["model"], patch_count=config["preprocessing"]["grid_size"] ** 2).eval()
    manifest = {"source_checkpoint_sha256": file_sha256(args.checkpoint), "initialization": "seed-matched, from scratch",
        "seed": config["seed"], "base_config": config, "subject_ids": people,
        "records": selected.drop(columns="file_path").to_dict("records"),
        "images": len(targets), "partitions_read": ["train"], "updates_per_variant": args.updates,
        "optimizer": {"name": "AdamW", "learning_rate": config["training"]["learning_rate"],
                      "weight_decay": config["training"]["weight_decay"], "clip": config["training"]["gradient_clip"]},
        "note": "Repeated balanced batch learnability controls, not held-out accuracy."}
    atomic_json(args.output / "manifest.json", manifest)
    rows = []
    for variant in args.variants:
        started = perf_counter()
        model, encoded = copy.deepcopy(original), None
        if variant == "shallow":
            model.config.update(H_cycles=1, L_cycles=1, refinement_steps=1)
        elif variant in {"h1", "l1", "k1"}:
            model.config[{"h1": "H_cycles", "l1": "L_cycles", "k1": "refinement_steps"}[variant]] = 1
        elif variant in {"raw_frozen_tokens", "normalized_frozen_tokens"}:
            with torch.no_grad():
                tokens = model.encode(patches)
                image = tokens[:, 1:]
                mean, scale = image.mean((0, 1), keepdim=True), image.std((0, 1), correction=0, keepdim=True).clamp_min(1e-6)
                # Unit-scale inputs like official token embeddings. Fit only the
                # eight selected training images; encoder/readout stay frozen.
                encoded = (torch.cat((tokens[:, :1], (image - mean) / scale), 1)
                           if variant == "normalized_frozen_tokens" else tokens).detach()
            for p in model.encoder.parameters():
                p.requires_grad_(False)
            model.readout.requires_grad_(False)
        elif variant.startswith("cnn_"):
            model = EncoderClassifier(original, moments=variant == "cnn_mean_std")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
        rows.append({"variant": variant, "updates": 0, "elapsed_seconds": 0., **measure(model, patches, targets, encoded)})
        step = 0
        while step < args.updates:
            recursive = isinstance(model, VisionTRM)
            if recursive:
                state = model.initial_state(len(targets))
            refinements = model.config["refinement_steps"] if recursive else 1
            for _ in range(refinements):
                optimizer.zero_grad(set_to_none=True)
                if recursive:
                    tokens = encoded if encoded is not None else model.encode(patches)
                    state, logits = model.refine_once(tokens, state)
                else:
                    logits = model(patches)
                loss = nn.functional.cross_entropy(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(variant)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step in {4, 16, 40, 100, 200, args.updates}:
                    row = {"variant": variant, "updates": step, "elapsed_seconds": perf_counter() - started,
                           **measure(model, patches, targets, encoded)}
                    rows.append(row)
                    print(row, flush=True)
                    pd.DataFrame(rows).to_csv(args.output / "history.csv", index=False)
                if step == args.updates:
                    break
        # This is a separate diagnostic checkpoint, never a production model.
        torch.save(model.state_dict(), args.output / f"{variant}.pt")
    atomic_json(args.output / "summary.json", {v: [r for r in rows if r["variant"] == v][-1] for v in args.variants})


if __name__ == "__main__":
    main()
