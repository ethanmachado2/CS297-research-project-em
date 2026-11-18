"""Five subject-grouped folds of the existing recursion-versus-depth experiment.

python -m src.recursion_crossval --reference-run outputs/PREVIOUS_RECURSION_RUN \
    --output outputs/FRESH_CROSSVAL_RUN

All fold/seed/model checkpoints are fixed before any out-of-fold predictions.
Images may be training examples in other folds; only each fit's own train and
validation partitions enter its optimization and checkpoint selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from .image_quality_pipeline import build_dataset
from .moment_ablation import train_one
from .recursion_ablation import SCHEMA, VARIANTS, evaluate_selected, make_depth_model
from .run import PROJECT, snapshot_source
from .trm_data import (create_subject_split, file_sha256, load_image_patches,
                       manifest_digest, partition_frame, validate_subject_split)
from .trm_training import atomic_json, environment_info


def grouped_splits(frame, root, *, seed=42, folds=5, validation_subjects=5):
    subjects = frame.subject_id.nunique()
    if type(folds) is not int or folds < 2 or subjects % folds:
        raise ValueError("Subject count must divide evenly into at least two folds")
    test_subjects = subjects // folds
    train_subjects = subjects - test_subjects - validation_subjects
    splits = [create_subject_split(frame, seed, project_root=root,
        counts=[train_subjects, validation_subjects, test_subjects], fold_index=index, folds=folds)
        for index in range(folds)]
    tests = Counter(subject for split in splits for subject in split["partitions"]["test"]["subject_ids"])
    if set(tests) != set(frame.subject_id.astype(str)) or set(tests.values()) != {1}:
        raise AssertionError("Each subject must appear in exactly one outer test fold")
    for split in splits:
        groups = [set(value["subject_ids"]) for value in split["partitions"].values()]
        if sum(map(len, groups)) != len(set.union(*groups)):
            raise AssertionError("Subject overlap inside a fold")
    return splits


def cached_data(frame, preprocessing, cache):
    """Cache deterministic crops only; partition membership controls all access."""
    for path in frame.file_path:
        if path not in cache:
            cache[path] = load_image_patches(path, preprocessing)
    return TensorDataset(torch.stack([cache[path] for path in frame.file_path]),
                         torch.tensor(frame.severity.astype(int).tolist(), dtype=torch.long))


def completed_fit(directory, protocol, variant, seed):
    """Accept only a fully finished fit with unchanged selected checkpoint."""
    summary = directory / "training_summary.json"
    if not summary.exists():
        return None
    row = json.loads(summary.read_text())
    checkpoint = torch.load(directory / "best.pt", weights_only=True)
    expected_updates = protocol["epochs"] * ((protocol["training_images"] + protocol["batch_size"] - 1) // protocol["batch_size"])
    if (row["variant"] != variant or row["seed"] != seed or
            checkpoint["variant"] != variant or checkpoint["seed"] != seed or
            checkpoint["schema_version"] != SCHEMA or checkpoint["protocol"] != protocol or
            row["optimizer_updates"] != expected_updates or
            file_sha256(directory / "best.pt") != row["checkpoint_sha256"]):
        raise ValueError(f"Completed fit identity, protocol or checkpoint changed: {directory}")
    return row


def aggregate_oof(output, frame, seeds, folds):
    """Pool held-out images within each seed; folds are not independent repeats."""
    tables = {}
    for name in ("prefix_predictions", "prefix_metrics", "image_transitions", "transition_metrics", "prefix_timing"):
        parts = [pd.read_csv(output / f"fold_{index + 1:02d}" / f"{name}.csv").assign(fold=index + 1)
                 for index in range(folds)]
        tables[name] = pd.concat(parts, ignore_index=True)
    predictions = tables["prefix_predictions"]
    transitions = tables["image_transitions"]
    all_images = set(frame.file_path)
    expected_groups = {(variant, seed, step) for variant in VARIANTS for seed in seeds
                       for step in ((1,) if variant == "shared_1" else (1, 2, 4))}
    actual_groups = set()
    metrics = []
    for (variant, seed, step), group in predictions.groupby(["variant", "seed", "step"], sort=False):
        actual_groups.add((variant, seed, step))
        if len(group) != len(frame) or group.image_path.duplicated().any() or set(group.image_path) != all_images:
            raise ValueError("Each image must have exactly one held-out prediction per variant/seed/prefix")
        fold_metrics = tables["prefix_metrics"]
        parameters = fold_metrics.loc[(fold_metrics.variant == variant) & (fold_metrics.seed == seed), "parameters"].unique()
        if len(parameters) != 1:
            raise ValueError("Model size differs across folds")
        metrics.append({"variant": variant, "seed": int(seed), "step": int(step),
            "images": len(group), "subjects": group.subject_id.nunique(),
            "severity_accuracy": group.severity_correct.mean(), "cross_entropy": group.cross_entropy.mean(),
            "severity_mae": group.severity_absolute_error.mean(), "routing_errors": int(group.routing_error.sum()),
            "route_false_positives": int(((group.predicted_route == 1) & (group.true_route == 0)).sum()),
            "route_false_negatives": int(((group.predicted_route == 0) & (group.true_route == 1)).sum()),
            "parameters": int(parameters[0])})
    if actual_groups != expected_groups:
        raise ValueError("Missing or unexpected held-out model/seed/prefix groups")
    pair_metrics = []
    for (variant, seed, early, late), group in transitions.groupby(["variant", "seed", "from_step", "to_step"], sort=False):
        if len(group) != len(frame) or group.image_path.duplicated().any() or set(group.image_path) != all_images:
            raise ValueError("Each image must appear once per held-out transition")
        early_correct = group.earlier_prediction == group.true_severity
        late_correct = group.later_prediction == group.true_severity
        corrected, regressed = ~early_correct & late_correct, early_correct & ~late_correct
        if not (group.corrected.to_numpy() == corrected.to_numpy()).all() or not (group.regressed.to_numpy() == regressed.to_numpy()).all():
            raise ValueError("Saved corrections/regressions do not match predictions")
        pair_metrics.append({"variant": variant, "seed": int(seed), "from_step": int(early), "to_step": int(late),
            "images": len(group), "corrections": int(corrected.sum()), "regressions": int(regressed.sum()),
            "net_corrections": int(corrected.sum() - regressed.sum()),
            "unchanged_correct": int((early_correct & late_correct).sum()),
            "unchanged_incorrect": int((~early_correct & ~late_correct).sum()),
            "routing_corrections": int(group.routing_corrected.sum()),
            "routing_regressions": int(group.routing_regressed.sum()),
            "cross_entropy_delta": group.cross_entropy_delta.mean(),
            "severity_mae_delta": group.severity_absolute_error_delta.mean()})
    expected_pairs = {(variant, seed, early, late) for variant in ("shared_4", "untied_4")
                      for seed in seeds for early, late in ((1, 2), (1, 4), (2, 4))}
    if set((r["variant"], r["seed"], r["from_step"], r["to_step"]) for r in pair_metrics) != expected_pairs:
        raise ValueError("Missing held-out transition groups")
    metrics, pairs = pd.DataFrame(metrics), pd.DataFrame(pair_metrics)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    transitions.to_csv(output / "oof_image_transitions.csv", index=False)
    metrics.to_csv(output / "oof_prefix_metrics.csv", index=False)
    pairs.to_csv(output / "oof_transition_metrics.csv", index=False)
    for name in ("prefix_metrics", "transition_metrics", "prefix_timing"):
        tables[name].to_csv(output / f"fold_{name}.csv", index=False)
    summary = metrics.groupby(["variant", "step"], sort=False).agg(
        seeds=("seed", "count"), images_per_seed=("images", "first"),
        accuracy_mean=("severity_accuracy", "mean"), accuracy_std=("severity_accuracy", "std"),
        cross_entropy_mean=("cross_entropy", "mean"), severity_mae_mean=("severity_mae", "mean"),
        routing_errors_mean=("routing_errors", "mean"), parameters=("parameters", "first"))
    summary.to_csv(output / "comparison_summary.csv")
    pairs.groupby(["variant", "from_step", "to_step"], sort=False).agg(
        seeds=("seed", "count"), corrections_mean=("corrections", "mean"),
        regressions_mean=("regressions", "mean"), net_corrections_mean=("net_corrections", "mean"),
        cross_entropy_delta_mean=("cross_entropy_delta", "mean"),
        severity_mae_delta_mean=("severity_mae_delta", "mean"),
        routing_corrections_mean=("routing_corrections", "mean"),
        routing_regressions_mean=("routing_regressions", "mean")).to_csv(output / "transition_summary.csv")
    atomic_json(output / "coverage.json", {"folds": folds, "unique_images": len(frame),
        "unique_subjects": int(frame.subject_id.nunique()), "initialization_seeds": list(seeds),
        "each_image_tested_once_per_variant_seed_prefix": True,
        "all_predefined_checkpoints_fixed_before_any_oof_prediction": True,
        "folds_have_overlapping_training_sets": True,
        "interpretation": "Exploratory cross-validation of an already inspected dataset; seeds are repeated fits, not new subjects"})
    print(summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, help="Previous recursion_ablation run defining the protocol")
    parser.add_argument("--dataset-root", type=Path, help="Optional relocated copy with identical image hashes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--resume", action="store_true", help="Reuse completed fits; restart interrupted fits from scratch")
    args = parser.parse_args()
    if not args.resume and args.reference_run is None:
        parser.error("A new run requires --reference-run")
    if args.resume and any(value is not None for value in (args.reference_run, args.dataset_root, args.epochs, args.seeds, args.threads)):
        parser.error("--resume uses the saved protocol; do not override it")
    output = args.output.resolve()
    started = perf_counter()
    if args.resume:
        protocol = json.loads((output / "protocol.json").read_text())
        for relative, expected in json.loads((output / "source_hashes.json").read_text()).items():
            if file_sha256(PROJECT / relative) != expected:
                raise ValueError(f"Source changed since this run started: {relative}")
        status = json.loads((output / "run.json").read_text())
        if status["status"] == "completed":
            print(f"Already completed: {output}", flush=True)
            return
        splits = [json.loads((output / f"fold_{i + 1:02d}/split.json").read_text()) for i in range(5)]
    else:
        reference = args.reference_run.resolve()
        base = json.loads((reference / "protocol.json").read_text())
        reference_split = json.loads((reference / "split.json").read_text())
        if base["checkpoint_schema"] != SCHEMA or tuple(base["variants"]) != VARIANTS or base["device"] != "cpu" or base["precision"] != "float32":
            raise ValueError("Reference must be the existing CPU FP32 three-variant recursion comparison")
        protocol = copy.deepcopy(base)
        protocol.update(epochs=args.epochs if args.epochs is not None else base["epochs"],
            seeds=args.seeds if args.seeds is not None else base["seeds"],
            threads=args.threads if args.threads is not None else base["threads"],
            dataset_root=str((args.dataset_root or Path(base["dataset_root"])).resolve()),
            started_at=datetime.now(timezone.utc).isoformat(),
            test_status="Exploratory five-fold subject-grouped cross-validation of the previously inspected dataset",
            cross_validation={"folds": 5, "split_seed": reference_split["seed"],
                "validation_subjects": reference_split["counts"][1], "reference_run": str(reference),
                "reference_protocol_sha256": file_sha256(reference / "protocol.json"),
                "selection": "Separate validation set inside each outer training pool; no tuning from outer test results"})
        if (min(protocol["epochs"], protocol["threads"]) < 1 or not protocol["seeds"] or
                len(set(protocol["seeds"])) != len(protocol["seeds"]) or
                any(not 0 <= seed < 2**32 for seed in protocol["seeds"])):
            parser.error("Positive epochs/threads and distinct nonnegative 32-bit seeds required")
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "protocol.json", protocol)
        atomic_json(output / "source_hashes.json", snapshot_source(output))
        frame = build_dataset(Path(protocol["dataset_root"]))
        validate_subject_split(reference_split, frame, Path(protocol["dataset_root"]))
        splits = grouped_splits(frame, Path(protocol["dataset_root"]), seed=reference_split["seed"],
                                validation_subjects=reference_split["counts"][1])
        for index, split in enumerate(splits):
            directory = output / f"fold_{index + 1:02d}"
            directory.mkdir()
            atomic_json(directory / "split.json", split)
            fold_protocol = {**protocol, "fold_index": index,
                "split_digest": manifest_digest(split), "training_images": len(split["partitions"]["train"]["records"])}
            atomic_json(directory / "protocol.json", fold_protocol)
        status = {"status": "preparing", "completed_training_runs": 0,
                  "planned_runs": 5 * len(VARIANTS) * len(protocol["seeds"])}
    torch.set_num_threads(protocol["threads"])
    atomic_json(output / "environment.json", environment_info(torch.device("cpu"), "float32"))
    atomic_json(output / "run.json", status)
    elapsed_before = status.get("elapsed_seconds", 0.0)
    try:
        root = Path(protocol["dataset_root"])
        frame = build_dataset(root)
        for split in splits:
            validate_subject_split(split, frame, root)
        expected = grouped_splits(frame, root, seed=protocol["cross_validation"]["split_seed"],
                                  validation_subjects=protocol["cross_validation"]["validation_subjects"])
        if splits != expected:
            raise ValueError("Saved folds changed from the predefined grouped split")
        factory = partial(make_depth_model, model_config=protocol["model_config"],
                          patch_count=protocol["preprocessing"]["grid_size"] ** 2)
        cache, selected, digests = {}, [], {}
        status["completed_training_runs"] = 0
        for index, split in enumerate(splits):
            directory = output / f"fold_{index + 1:02d}"
            fold_protocol = json.loads((directory / "protocol.json").read_text())
            expected_protocol = {**protocol, "fold_index": index, "split_digest": manifest_digest(split),
                                 "training_images": len(split["partitions"]["train"]["records"])}
            if fold_protocol != expected_protocol:
                raise ValueError("Saved fold protocol changed")
            print(f"FOLD {index + 1}/5: preparing training/validation partitions", flush=True)
            train = cached_data(partition_frame(split, "train", root), protocol["preprocessing"], cache)
            validation = cached_data(partition_frame(split, "validation", root), protocol["preprocessing"], cache)
            rows = []
            for seed in protocol["seeds"]:
                for variant in VARIANTS:
                    status.update(status="training", current_fold=index + 1, current_variant=variant, current_seed=seed)
                    atomic_json(output / "run.json", status)
                    fit = directory / f"{variant}_seed{seed}"
                    row = completed_fit(fit, fold_protocol, variant, seed) if args.resume else None
                    if row is None:
                        if fit.exists():
                            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                            fit.rename(directory / f"interrupted_{fit.name}_{timestamp}")
                        row = train_one(variant, seed, train, validation, fit, fold_protocol, model_factory=factory)
                    else:
                        print(f"Reusing completed fold={index + 1} {variant} seed={seed}", flush=True)
                    if seed in digests and digests[seed] != row["initial_feature_digest"]:
                        raise AssertionError("Initial moment encoder/positions differ across fits")
                    digests[seed] = row["initial_feature_digest"]
                    rows.append(row)
                    selected.append({**row, "fold": index + 1})
                    atomic_json(directory / "selected_checkpoints.json", rows)
                    atomic_json(output / "selected_checkpoints.json", selected)
                    status["completed_training_runs"] += 1
                    status["elapsed_seconds"] = elapsed_before + perf_counter() - started
                    atomic_json(output / "run.json", status)
            del train, validation
        if len(selected) != status["planned_runs"]:
            raise AssertionError("All predefined checkpoints must be fixed before any test inference")
        status.update(status="evaluating", all_checkpoints_selected_before_oof_predictions=True)
        atomic_json(output / "run.json", status)
        for index, split in enumerate(splits):
            directory = output / f"fold_{index + 1:02d}"
            fold_protocol = json.loads((directory / "protocol.json").read_text())
            rows = json.loads((directory / "selected_checkpoints.json").read_text())
            test = cached_data(partition_frame(split, "test", root), protocol["preprocessing"], cache)
            evaluate_selected(directory, fold_protocol, split, frame, rows, test_data=test)
        aggregate_oof(output, frame, protocol["seeds"], 5)
        status.update(status="completed", unique_test_images=len(frame),
            unique_test_subjects=int(frame.subject_id.nunique()),
            initial_features_identical_within_seed=True,
            elapsed_seconds=elapsed_before + perf_counter() - started)
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}", elapsed_seconds=elapsed_before + perf_counter() - started)
        raise
    finally:
        atomic_json(output / "run.json", status)


if __name__ == "__main__":
    main()
