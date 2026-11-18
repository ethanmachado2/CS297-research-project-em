"""Controlled comparison of ViTRM encoders and a direct classifier.

The comparison keeps the recursive core, subject split, optimizer protocol,
and training budget fixed while changing only the visual encoder / presence of
recursive reasoning:

* ``linear_vitrm``: per-cell linear projection followed by ViTRM.
* ``moment_vitrm``: CNN/RGB-moment encoder followed by ViTRM.
* ``moment_direct``: the same CNN/RGB-moment encoder and positional features,
  mean pooled and sent directly to a classifier.

This is intentionally separate from the historical moment_ablation runner so
that the three scientific comparison variants have unambiguous names and a
small, auditable protocol.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .image_quality_pipeline import build_dataset
from .trm_data import (FaceSeverityDataset, create_subject_split, load_subject_split,
                       partition_frame, validate_image_sizes)
from .trm_training import seed_everything
from .vitrm_model import MomentEncoder, VisionViTRM

VARIANTS = ("linear_vitrm", "moment_vitrm", "moment_direct")


class LinearCellEncoder(nn.Module):
    """ViT-style linear cell projection with no learned visual feature stack."""

    def __init__(self, width: int, grid: int, patch_size: int):
        super().__init__()
        if patch_size % grid:
            raise ValueError("patch_size must be divisible by token_grid")
        cell = patch_size // grid
        self.grid = grid
        self.patch_size = patch_size
        self.projection = nn.Linear(3 * cell * cell, width)

    def forward(self, patches):
        batch, crops, channels, height, width = patches.shape
        if channels != 3 or height != self.patch_size or width != self.patch_size:
            raise ValueError("LinearCellEncoder received unexpected patch shape")
        g, cell = self.grid, self.patch_size // self.grid
        cells = patches.reshape(batch, crops, channels, g, cell, g, cell)
        cells = cells.permute(0, 1, 3, 5, 2, 4, 6).reshape(
            batch, crops * g * g, channels * cell * cell)
        return self.projection(cells)


class DirectMomentClassifier(nn.Module):
    """Direct classifier using exactly the ViTRM moment encoder and positions."""

    def __init__(self, config, *, patch_count):
        super().__init__()
        width, grid = config["hidden_size"], config["token_grid"]
        self.config = dict(config)
        self.patch_count = patch_count
        self.encoder = MomentEncoder(width, grid)
        self.crop_position = nn.Parameter(torch.empty(1, patch_count, 1, width))
        self.cell_position = nn.Parameter(torch.empty(1, 1, grid ** 2, width))
        self.output_norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 4)
        for parameter in (self.crop_position, self.cell_position):
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(self, patches):
        tokens = self.encoder(patches)
        positions = (self.crop_position + self.cell_position).flatten(1, 2)
        return self.classifier(self.output_norm((tokens + positions).mean(dim=1)))

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "encoder": sum(p.numel() for p in self.encoder.parameters()),
                "reasoning": 0, "classifier": sum(p.numel() for p in self.classifier.parameters()),
                "states_and_positions": sum(p.numel() for p in (self.crop_position, self.cell_position))}


def make_model(variant, config):
    patch_count = config["preprocessing"]["grid_size"] ** 2
    model_config = config["model"]
    if variant == "linear_vitrm":
        model = VisionViTRM(model_config, patch_count=patch_count)
        model.encoder = LinearCellEncoder(model_config["hidden_size"],
                                          model_config["token_grid"],
                                          config["preprocessing"]["patch_size"])
        return model
    if variant == "moment_vitrm":
        return VisionViTRM(model_config, patch_count=patch_count)
    if variant == "moment_direct":
        return DirectMomentClassifier(model_config, patch_count=patch_count)
    raise ValueError(f"Unknown comparison variant: {variant}")


def load_pixels(frame, preprocessing):
    dataset = FaceSeverityDataset(frame, preprocessing)
    pixels, labels = zip(*(dataset[index] for index in range(len(dataset))))
    return torch.stack(pixels), torch.tensor(labels, dtype=torch.long)


def autocast_context(device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)


def logits_for(model, pixels):
    if isinstance(model, VisionViTRM):
        steps = model.forward_steps(pixels)
        return steps, steps[:, -1]
    logits = model(pixels)
    return logits[:, None], logits


def mean_loss(step_logits, labels):
    return F.cross_entropy(step_logits.flatten(0, 1),
                           labels[:, None].expand(-1, step_logits.shape[1]).reshape(-1))


def metrics(logits, labels):
    predictions = logits.argmax(-1).cpu()
    labels = labels.cpu()
    matrix = torch.zeros(4, 4, dtype=torch.long)
    for truth, guess in zip(labels.tolist(), predictions.tolist()):
        matrix[truth, guess] += 1
    f1 = []
    for index in range(4):
        denom = matrix[index].sum().item() + matrix[:, index].sum().item()
        f1.append(2 * matrix[index, index].item() / denom if denom else 0.0)
    return {"accuracy": float((predictions == labels).float().mean()),
            "macro_f1": float(sum(f1) / 4),
            "mae": float((predictions - labels).abs().float().mean())}


def evaluate(model, pixels, labels, device):
    model.eval()
    with torch.inference_mode(), autocast_context(device):
        steps, final = logits_for(model, pixels.to(device))
    result = metrics(final, labels)
    result.update({"validation_steps": int(steps.shape[1]),
                   "parameter_count": sum(p.numel() for p in model.parameters())})
    return result


def train_one(variant, seed, config, train_pixels, train_labels, val_pixels, val_labels,
              output, device, epochs, batch_size):
    seed_everything(seed)
    model = make_model(variant, config).to(device)
    model.precision = "float16" if device.type == "cuda" else "float32"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loader = DataLoader(TensorDataset(train_pixels, train_labels), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_loss, best_epoch, stale, updates = math.inf, 0, 0, 0
    history = []
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for pixels, labels in loader:
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                step_logits, _ = logits_for(model, pixels)
                loss = mean_loss(step_logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            updates += 1
            train_loss += loss.item() * len(labels)
        model.eval()
        with torch.inference_mode(), autocast_context(device):
            val_steps, val_logits = logits_for(model, val_pixels.to(device))
            val_loss = mean_loss(val_steps, val_labels.to(device)).item()
        val_metrics = metrics(val_logits, val_labels)
        row = {"epoch": epoch, "train_loss": train_loss / len(train_labels),
               "validation_loss": val_loss, "optimizer_updates": updates, **val_metrics}
        for index, step_logits in enumerate(val_steps.unbind(dim=1), 1):
            row[f"step_{index}_accuracy"] = metrics(step_logits, val_labels)["accuracy"]
        history.append(row)
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({"schema_version": 1, "variant": variant, "seed": seed,
                        "config": copy.deepcopy(config), "model_state": model.state_dict(),
                        "validation_loss": val_loss, "epoch": epoch}, output / "best.pt")
        else:
            stale += 1
        if stale >= config["training"]["patience"]:
            break
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model = make_model(variant, config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    result = evaluate(model, val_pixels, val_labels, device)
    result.update({"variant": variant, "seed": seed, "best_epoch": best_epoch,
                   "best_validation_loss": best_loss, "optimizer_updates": updates,
                   "checkpoint": str((output / "best.pt").resolve())})
    return model, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-split", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/vitrm.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    config = json.loads(args.config.read_text())
    config["training"]["device"] = args.device
    config["training"]["precision"] = "float16" if args.device == "cuda" else "float32"
    frame = build_dataset(args.dataset_root)
    validate_image_sizes(frame, config["preprocessing"])
    args.output.mkdir(parents=True, exist_ok=False)
    if args.reference_split:
        split = load_subject_split(args.reference_split, frame, args.dataset_root)
    else:
        split = create_subject_split(frame, config["split"]["seed"], project_root=args.dataset_root,
                                     counts=config["split"]["counts"])
    (args.output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    train_frame = partition_frame(split, "train", args.dataset_root)
    val_frame = partition_frame(split, "validation", args.dataset_root)
    train_pixels, train_labels = load_pixels(train_frame, config["preprocessing"])
    val_pixels, val_labels = load_pixels(val_frame, config["preprocessing"])
    rows = []
    trained = []
    for seed in args.seeds:
        for variant in VARIANTS:
            print(f"Training {variant}, seed={seed}", flush=True)
            model, row = train_one(variant, seed, config, train_pixels, train_labels,
                                   val_pixels, val_labels, args.output / variant / f"seed_{seed}",
                                   device, args.epochs, args.batch_size)
            rows.append(row)
            trained.append((variant, seed, row["checkpoint"]))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    # Do not touch test pixels until every model has selected its checkpoint.
    test_frame = partition_frame(split, "test", args.dataset_root)
    test_pixels, test_labels = load_pixels(test_frame, config["preprocessing"])
    for variant, seed, checkpoint_path in trained:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = make_model(variant, config)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        test_metrics = evaluate(model, test_pixels, test_labels, device)
        row = next(item for item in rows if item["variant"] == variant and item["seed"] == seed)
        row.update({f"test_{key}": value for key, value in test_metrics.items()
                    if key in {"accuracy", "macro_f1", "mae"}})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "comparison_runs.csv", index=False)
    result.groupby("variant", as_index=False).agg(
        validation_accuracy_mean=("accuracy", "mean"),
        validation_accuracy_std=("accuracy", "std"),
        test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", "std"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        test_mae_mean=("test_mae", "mean"),
    ).to_csv(args.output / "comparison_summary.csv", index=False)
    (args.output / "protocol.json").write_text(json.dumps({
        "variants": VARIANTS, "seeds": args.seeds, "epochs": args.epochs,
        "batch_size": args.batch_size, "device": args.device,
        "test_loaded_after_checkpoint_selection": True,
        "linear_definition": "3x32x32 cell flattened then one Linear(width)",
        "direct_definition": "moment tokens plus positions mean-pooled then LayerNorm/Linear",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
