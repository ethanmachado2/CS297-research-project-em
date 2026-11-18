"""Controlled CPU screening of moment features, head choice, and recurrence.

python -m src.moment_ablation --dataset-root PATH --output FRESH_DIRECTORY

All models start with identical MomentEncoder weights and position embeddings
within each seed, then learn end to end independently. The old TRM keeps its
original H/L/outer depths, additive injection, scaling and truncated gradients;
mean-loss accumulation replaces its per-refinement optimizer updates so every
model receives the same number of batches, image exposures and optimizer steps.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .image_quality_pipeline import build_dataset, run_inference
from .run import PROJECT, snapshot_source
from .trm_adapter import export_refinements, severity_prediction
from .trm_data import (create_subject_split, file_sha256, load_image_patches,
                       manifest_digest, partition_frame, validate_subject_split)
from .trm_model import UPSTREAM_SHA, VisionTRM
from .trm_reporting import export_saved_run
from .trm_training import atomic_checkpoint, atomic_json, environment_info, seed_everything
from .vitrm_model import VisionViTRM

VARIANTS = ("direct_linear", "direct_mlp", "old_trm", "vitrm_1", "vitrm_2")
OLD_CONFIG = dict(hidden_size=128, num_heads=4, L_layers=2, expansion=4.0,
                  H_cycles=2, L_cycles=2, refinement_steps=4)


class MomentFeatures(nn.Module):
    """Take exactly the initialized encoder and positions from a ViTRM."""
    def __init__(self, base):
        super().__init__()
        self.encoder = base.encoder
        self.crop_position = base.crop_position
        self.cell_position = base.cell_position
        self.patch_count = base.patch_count

    def forward(self, patches):
        if patches.ndim != 5 or patches.shape[1:3] != (self.patch_count, 3):
            raise ValueError("Unexpected native crop shape")
        tokens = self.encoder(patches)
        return tokens + (self.crop_position + self.cell_position).flatten(1, 2).to(tokens.dtype)


class MomentDirect(nn.Module):
    def __init__(self, base, *, mlp=False):
        super().__init__()
        width = base.config["hidden_size"]
        self.features = MomentFeatures(base)
        self.norm = copy.deepcopy(base.output_norm)
        self.head = (nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                   copy.deepcopy(base.classifier))
                     if mlp else copy.deepcopy(base.classifier))
        self.config = {"refinement_steps": 1}

    def forward_steps(self, patches):
        return self.head(self.norm(self.features(patches).mean(dim=1)))[:, None]

    def forward(self, patches):
        return self.forward_steps(patches)[:, -1]


class MomentOldTRM(nn.Module):
    def __init__(self, base, config=None):
        super().__init__()
        self.features = MomentFeatures(base)
        token_count = base.patch_count * base.config["token_grid"] ** 2
        self.config = dict(config or OLD_CONFIG)
        self.core = VisionTRM(self.config, patch_count=token_count)
        # The discarded original encoder must not inflate counts or receive updates.
        self.core.encoder = nn.Identity()

    def encode(self, patches):
        image = self.features(patches)
        return torch.cat((self.core.readout.expand(len(patches), -1, -1), image), dim=1) * math.sqrt(self.config["hidden_size"])

    def forward_steps(self, patches):
        encoded = self.encode(patches)
        state = self.core.initial_state(len(patches), patches.device)
        outputs = []
        for _ in range(self.config["refinement_steps"]):
            state, logits = self.core.refine_once(encoded, state)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)

    def forward(self, patches):
        return self.forward_steps(patches)[:, -1]


def make_model(variant, seed, model_config=None, *, patch_count=9, old_config=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown ablation: {variant}")
    seed_everything(seed)
    config = dict(model_config or json.loads((PROJECT / "configs/vitrm.json").read_text())["model"])
    config["refinement_steps"] = 1 if variant == "vitrm_1" else 2
    base = VisionViTRM(config, patch_count=patch_count)
    if variant.startswith("direct_"):
        return MomentDirect(base, mlp=variant == "direct_mlp")
    if variant == "old_trm":
        return MomentOldTRM(base, old_config)
    return base


def feature_digest(model):
    source = model if isinstance(model, VisionViTRM) else model.features
    digest = hashlib.sha256()
    for key, tensor in source.state_dict().items():
        if key.startswith("encoder.") or key in {"crop_position", "cell_position"}:
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def mean_supervised_loss(logits, targets):
    return nn.functional.cross_entropy(logits.flatten(0, 1),
        targets[:, None].expand(-1, logits.shape[1]).reshape(-1))


def score(model, data, batch_size=8):
    model.eval()
    outputs, truths = [], []
    with torch.inference_mode():
        for pixels, targets in DataLoader(data, batch_size=batch_size):
            steps = model.forward_steps(pixels)
            if not torch.isfinite(steps).all():
                raise FloatingPointError("Non-finite evaluation logits")
            outputs.append(steps)
            truths.append(targets)
    logits, labels = torch.cat(outputs), torch.cat(truths)
    final = logits[:, -1]
    return {"loss": nn.functional.cross_entropy(final, labels).item(),
            "accuracy": (final.argmax(-1) == labels).float().mean().item(),
            "refinement_loss": [nn.functional.cross_entropy(step, labels).item()
                                for step in logits.unbind(dim=1)],
            "refinement_accuracy": [(step.argmax(-1) == labels).float().mean().item()
                                    for step in logits.unbind(dim=1)]}


def preload(frame, preprocess):
    pixels = torch.stack([load_image_patches(path, preprocess) for path in frame.file_path])
    targets = torch.tensor(frame.severity.astype(int).tolist(), dtype=torch.long)
    return TensorDataset(pixels, targets)


def train_one(variant, seed, train, validation, output, protocol, *, model_factory=None):
    output.mkdir(parents=True, exist_ok=False)
    model = (model_factory or make_model)(variant, seed)
    initial_digest = feature_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"],
                                  weight_decay=protocol["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=protocol["batch_size"], shuffle=True, generator=generator)
    history, best_loss, updates = [], float("inf"), 0
    started = perf_counter()
    for epoch in range(1, protocol["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for pixels, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = mean_supervised_loss(model.forward_steps(pixels), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total_loss += loss.item() * len(targets)
        validation_metrics = score(model, validation, protocol["batch_size"])
        if validation_metrics["loss"] < best_loss - 1e-6:
            best_loss = validation_metrics["loss"]
            atomic_checkpoint(output / "best.pt", {
                "schema_version": protocol.get("checkpoint_schema", "moment_ablation_v1"),
                "variant": variant, "seed": seed, "model_config": model.config,
                "epoch": epoch, "validation": validation_metrics, "protocol": protocol,
                "optimizer_updates": updates, "initial_feature_digest": initial_digest,
                "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}})
        history.append({"epoch": epoch, "training_loss": total_loss / len(train),
            "validation_loss": validation_metrics["loss"], "validation_accuracy": validation_metrics["accuracy"],
            "optimizer_updates": updates, "elapsed_seconds": perf_counter() - started})
        for step, (loss_value, accuracy) in enumerate(zip(validation_metrics["refinement_loss"],
                                                        validation_metrics["refinement_accuracy"]), 1):
            history[-1].update({f"validation_step_{step}_loss": loss_value,
                               f"validation_step_{step}_accuracy": accuracy})
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(f"{variant} seed={seed} epoch={epoch}/{protocol['epochs']}: "
              f"train={history[-1]['training_loss']:.4f} val={validation_metrics['loss']:.4f} "
              f"accuracy={validation_metrics['accuracy']:.3f}", flush=True)
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    result = {"variant": variant, "seed": seed, "selected_epoch": checkpoint["epoch"],
        "validation_accuracy": checkpoint["validation"]["accuracy"],
        "validation_loss": checkpoint["validation"]["loss"],
        "training_accuracy": score(model, train, protocol["batch_size"])["accuracy"],
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "optimizer_updates": updates, "initial_feature_digest": initial_digest,
        "training_seconds": perf_counter() - started, "checkpoint_sha256": file_sha256(output / "best.pt")}
    atomic_json(output / "training_summary.json", result)
    return result


class CachedTestAdapter:
    def __init__(self, model, frame, data, checkpoint, metadata):
        self.model = model.eval()
        self.pixels = {path: data.tensors[0][index:index + 1] for index, path in enumerate(frame.file_path)}
        self.refinement_records = []
        self.inference_config = {"uses_prompt": False, "training_regime": "supervised",
            "checkpoint": str(checkpoint), "device": "cpu", "dtype": "float32",
            "input": "cached native crops, identical for all models", **metadata}

    def generate(self, path, prompt):
        started = perf_counter()
        with torch.inference_mode():
            steps = self.model.forward_steps(self.pixels[str(path)])[0]
        elapsed = perf_counter() - started
        if not torch.isfinite(steps).all():
            raise FloatingPointError("Non-finite test logits")
        predictions, probabilities = steps.argmax(-1).tolist(), steps.softmax(-1).tolist()
        for index, (prediction, values) in enumerate(zip(predictions, probabilities)):
            self.refinement_records.append({"image_path": str(path), "step": index + 1,
                "predicted_severity": prediction, "model_seconds_all_steps": elapsed,
                **{f"probability_{i}": value for i, value in enumerate(values)}})
        return json.dumps(severity_prediction(predictions[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.epochs, args.threads) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive epochs/threads and distinct seeds required")
    torch.set_num_threads(args.threads)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    protocol = {"screening_only": True, "variants": list(VARIANTS), "seeds": args.seeds,
        "epochs": args.epochs, "batch_size": 8, "learning_rate": 0.0003, "weight_decay": 0.0001,
        "gradient_clip": 1.0, "precision": "float32", "device": "cpu", "threads": args.threads,
        "split_seed": 42, "counts": [35, 5, 10], "preprocessing": {"patch_size": 128, "grid_size": 3, "exif_transpose": False},
        "encoder": "Identical initial CNN/RGB spatial moments, projection and crop/cell positions within each seed; independently trained end to end",
        "old_trm": {**OLD_CONFIG, "token_count": 144, "upstream_sha": UPSTREAM_SHA,
                    "preserved": "sqrt(width) scale, readout token, additive injection, RoPE, fixed state initialization, H no_grad and outer detach boundaries"},
        "training": "All: mean CE over supervised outputs; ONE optimizer update per batch. Old TRM differs from historical per-refinement optimizer updates.",
        "direct_pooling": "Mean across the same 144 tokens; LayerNorm; linear head or width-128 GELU MLP",
        "selection": "Lowest final-step validation cross-entropy; train all predefined configurations before any test inference",
        "test_status": "Exploratory reuse of the same 40-image test cohort already observed in prior screening",
        "dataset_root": str(args.dataset_root.resolve()), "reference_split": str(args.reference_split.resolve()),
        "started_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(output / "protocol.json", protocol)
    status = {"status": "preparing", "completed_training_runs": 0, "planned_runs": len(VARIANTS) * len(args.seeds)}
    atomic_json(output / "run.json", status)
    try:
        atomic_json(output / "source_hashes.json", snapshot_source(output))
        atomic_json(output / "environment.json", environment_info(torch.device("cpu"), "float32"))
        frame = build_dataset(args.dataset_root)
        reference = json.loads(args.reference_split.read_text())
        split = create_subject_split(frame, 42, project_root=args.dataset_root, counts=[35, 5, 10])
        for name in split["partitions"]:
            if split["partitions"][name]["subject_ids"] != reference["partitions"][name]["subject_ids"]:
                raise ValueError("Subject split differs from reference")
            expected = {(row["subject_id"], row["severity"]): row["sha256"] for row in reference["partitions"][name]["records"]}
            actual = {(row["subject_id"], row["severity"]): row["sha256"] for row in split["partitions"][name]["records"]}
            if actual != expected:
                raise ValueError("Dataset bytes differ from reference")
        atomic_json(output / "split.json", split)
        train_frame = partition_frame(split, "train", args.dataset_root)
        val_frame = partition_frame(split, "validation", args.dataset_root)
        test_frame = partition_frame(split, "test", args.dataset_root)
        assert not (set(train_frame.subject_id) | set(val_frame.subject_id)) & set(test_frame.subject_id)
        print("Loading only 140 training and 20 validation images into RAM", flush=True)
        train, validation = preload(train_frame, protocol["preprocessing"]), preload(val_frame, protocol["preprocessing"])
        rows, digests = [], {}
        status["status"] = "training"
        for seed in args.seeds:
            for variant in VARIANTS:
                status.update(current_variant=variant, current_seed=seed)
                atomic_json(output / "run.json", status)
                row = train_one(variant, seed, train, validation, output / f"{variant}_seed{seed}", protocol)
                if seed in digests and digests[seed] != row["initial_feature_digest"]:
                    raise AssertionError("Initial moment features are not identical across models")
                digests[seed] = row["initial_feature_digest"]
                rows.append(row)
                status["completed_training_runs"] += 1
                atomic_json(output / "run.json", status)
        # Only after all models and checkpoints are fixed do test pixels enter memory.
        status["status"] = "evaluating"
        atomic_json(output / "run.json", status)
        validate_subject_split(split, frame, args.dataset_root)
        print("All training complete; loading 40 held-out test images once", flush=True)
        test = preload(test_frame, protocol["preprocessing"])
        for row in rows:
            directory = output / f"{row['variant']}_seed{row['seed']}"
            checkpoint = torch.load(directory / "best.pt", weights_only=True)
            model = make_model(row["variant"], row["seed"])
            model.load_state_dict(checkpoint["model_state"])
            adapter = CachedTestAdapter(model, test_frame, test, directory / "best.pt", row)
            run_inference(adapter, test_frame.file_path, labels=test_frame, output_dir=directory,
                experiment_number=2, run_number=1, model_name=row["variant"], inference_mode="supervised",
                batch_size=1, show_progress=False)
            run = directory / "experiment_002/run_001"
            report = export_saved_run(run)
            export_refinements(adapter, test_frame, run)
            row.update(test_severity_accuracy=report["metrics"]["severity"]["accuracy"],
                test_severity_macro_f1=report["metrics"]["severity"]["macro_f1"],
                test_severity_mae=report["metrics"]["severity"]["mae"],
                test_noise_accuracy=report["metrics"]["noise"]["accuracy"],
                test_route_accuracy=report["metrics"]["route"]["accuracy"],
                test_route_false_negatives=report["metrics"]["route"]["confusion_matrix"][1][0],
                test_route_false_positives=report["metrics"]["route"]["confusion_matrix"][0][1],
                test_valid=report["metrics"]["severity"]["evaluated"], test_failures=len(report["failures"]),
                model_seconds_per_image=float(pd.DataFrame(adapter.refinement_records).groupby("image_path").model_seconds_all_steps.first().mean()))
            pd.DataFrame(rows).to_csv(output / "comparison_runs.csv", index=False)
            print(f"TEST {row['variant']} seed={row['seed']}: severity={row['test_severity_accuracy']:.3f} "
                  f"route={row['test_route_accuracy']:.3f}", flush=True)
        results = pd.DataFrame(rows)
        grouped = results.groupby("variant", sort=False)
        summary = grouped.agg(runs=("seed", "count"),
            severity_mean=("test_severity_accuracy", "mean"), severity_std=("test_severity_accuracy", "std"),
            severity_min=("test_severity_accuracy", "min"), severity_max=("test_severity_accuracy", "max"),
            severity_macro_f1_mean=("test_severity_macro_f1", "mean"),
            route_mean=("test_route_accuracy", "mean"), route_std=("test_route_accuracy", "std"),
            noise_mean=("test_noise_accuracy", "mean"), validation_mean=("validation_accuracy", "mean"),
            parameters=("parameters", "first"), train_seconds_mean=("training_seconds", "mean"))
        summary.to_csv(output / "comparison_summary.csv")
        # Paired seeds are repeated fits on ONE split, not independent test cohorts.
        pivot = results.pivot(index="seed", columns="variant", values="test_severity_accuracy")
        differences = {name: (pivot["vitrm_2"] - pivot[name]).to_dict() for name in VARIANTS if name != "vitrm_2"}
        atomic_json(output / "paired_differences.json", differences)
        status.update(status="completed", elapsed_seconds=perf_counter() - started,
            split_digest=manifest_digest(split), test_images=40, test_subjects=10,
            initial_features_identical_within_seed=True, train_validation_test_subject_overlap=0)
        print(summary.to_string(), flush=True)
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(output / "run.json", status)


if __name__ == "__main__":
    main()
