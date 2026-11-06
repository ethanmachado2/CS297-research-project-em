"""Trace saved train/validation images; never open test image pixels.

Run: OMP_NUM_THREADS=2 .venv/bin/python scripts/diagnose_trm.py --output NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.trm_data import FaceSeverityDataset, file_sha256, partition_frame
from src.trm_diagnostics import (centroid_probe, compare_gradient_truncation, gradient_trace,
                                 ridge_probe, summarize_features, trace_features)
from src.trm_model import VisionTRM
from src.trm_training import atomic_json, load_checkpoint, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/trm_supervised_runs/20261024_implementation_validation/checkpoints/best.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    torch.set_num_threads(2)
    trained, checkpoint = load_checkpoint(args.checkpoint, "cpu")
    config, split = checkpoint["config"], checkpoint["split"]
    frames = {part: partition_frame(split, part, ROOT) for part in ("train", "validation")}
    for frame in frames.values():
        for row in frame.itertuples():
            if file_sha256(row.file_path) != row.sha256:
                raise ValueError(f"Image changed: {row.file_path}")
    # Loading a checkpoint constructs a model before overwriting it. Reset the
    # seed to reconstruct exactly the trainer's original initialization.
    seed_everything(config["seed"])
    initial = VisionTRM(config["model"], patch_count=config["preprocessing"]["grid_size"] ** 2).eval()
    features = {name: {part: {} for part in frames} for name in ("initial", "trained")}
    probabilities = {name: {part: [] for part in frames} for name in features}
    labels = {}
    gradient_batch = None
    for part, frame in frames.items():
        labels[part] = torch.tensor(frame.severity.astype(int).tolist())
        loader = DataLoader(FaceSeverityDataset(frame, config["preprocessing"]), batch_size=8, shuffle=False)
        for index, (patches, targets) in enumerate(loader):
            for name, model in (("initial", initial), ("trained", trained)):
                traced, probs = trace_features(model, patches)
                for stage, value in traced.items():
                    features[name][part].setdefault(stage, []).append(value)
                probabilities[name][part].append(torch.stack(probs, dim=1))
            # First two people, one image/class/person: balanced real gradient diagnostic.
            if part == "train" and gradient_batch is None:
                people = frame.subject_id.drop_duplicates().iloc[:2]
                selected = frame.loc[frame.subject_id.isin(people)]
                pairs = [FaceSeverityDataset(selected, config["preprocessing"])[i] for i in range(len(selected))]
                gradient_batch = (torch.stack([p[0] for p in pairs]), torch.tensor([p[1] for p in pairs]))
            print(f"{part}: {min((index + 1) * 8, len(frame))}/{len(frame)} images", flush=True)
    rows, probability_rows, centroid_results = [], [], {}
    for name in features:
        for part in frames:
            features[name][part] = {stage: torch.cat(chunks) for stage, chunks in features[name][part].items()}
            probs = torch.cat(probabilities[name][part])
            for step in range(probs.shape[1]):
                p = probs[:, step]
                prediction = p.argmax(-1)
                probability_rows.append({"model": name, "partition": part, "step": step + 1,
                    "accuracy": (prediction == labels[part]).double().mean().item(),
                    "probability_image_std_rms": p.std(0, correction=0).square().mean().sqrt().item(),
                    **{f"mean_probability_{i}": p[:, i].mean().item() for i in range(4)},
                    **{f"predicted_count_{i}": (prediction == i).sum().item() for i in range(4)}})
        for stage, train in features[name]["train"].items():
            val = features[name]["validation"][stage]
            probe = ridge_probe(train, labels["train"], val, labels["validation"])
            row = {"model": name, "stage": stage, **summarize_features(train, labels["train"]),
                   "probe_train_accuracy": probe["train"]["severity_accuracy"],
                   "probe_validation_accuracy": probe["validation"]["severity_accuracy"],
                   "probe_validation_route_accuracy": probe["validation"]["route_accuracy"]}
            rows.append(row)
        centroid_results[name] = {stage: centroid_probe(features[name]["train"][stage], labels["train"],
            features[name]["validation"][stage], labels["validation"])
            for stage in ("input/residual_std", "encoder/conv3_std", "encoder/projected_patch_mean", "outer/4_answer")
            if stage in features[name]["train"]}
    pd.DataFrame(rows).to_csv(args.output / "stages.csv", index=False)
    pd.DataFrame(probability_rows).to_csv(args.output / "probabilities.csv", index=False)
    gradients = {name: gradient_trace(model, *gradient_batch) for name, model in (("initial", initial), ("trained", trained))}
    atomic_json(args.output / "gradients.json", gradients)
    atomic_json(args.output / "truncation_check.json", compare_gradient_truncation(trained, *gradient_batch))
    atomic_json(args.output / "centroid_probes.json", centroid_results)
    atomic_json(args.output / "manifest.json", {"checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint), "split_digest": checkpoint["split_digest"],
        "partitions": {part: {"images": len(frame), "people": frame.subject_id.nunique()} for part, frame in frames.items()},
        "test_pixels_read": False, "seed": config["seed"], "probe": "train-standardized ridge, alpha=1; diagnostic only",
        "device": "cpu", "threads": 2, "torch": str(torch.__version__),
        "sources": {str(path.relative_to(ROOT)): file_sha256(path) for path in
                    (Path(__file__), ROOT / "src/trm_diagnostics.py", ROOT / "src/trm_model.py")},
        "elapsed_seconds": perf_counter() - started})
    # Small frozen tensors allow additional analysis without reopening any images.
    torch.save({"features": features, "labels": labels}, args.output / "features.pt")
    print(f"Saved trace to {args.output}", flush=True)


if __name__ == "__main__":
    main()
