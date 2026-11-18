"""Compare trained ViTRM prefixes with ordinary, untied Transformer depth.

python -m src.recursion_ablation --dataset-root PATH --reference-split PATH \
    --output FRESH_DIRECTORY

One four-step checkpoint supplies steps 1, 2 and 4. Each image is encoded once;
all reported prefixes come from that single unroll. The ordinary-depth control
has the same computation graph but independent weights at EVERY Transformer
call, including the memory and answer phases. It is an untied feed-forward
unrolling, not a conventional CLS-token ViT. Encoder, initial states, positions,
normalization and classification head are shared across its depth.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from functools import partial
from itertools import combinations
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .image_quality_pipeline import build_dataset
from .moment_ablation import preload, train_one
from .run import PROJECT, snapshot_source
from .trm_data import (file_sha256, manifest_digest, partition_frame,
                       validate_subject_split)
from .trm_training import atomic_json, environment_info, seed_everything
from .vitrm_model import VisionViTRM

VARIANTS = ("shared_1", "shared_4", "untied_4")
DEPTHS = (1, 2, 4)
SCHEMA = "recursion_depth_ablation_v1"


class UntiedVisionDepth(VisionViTRM):
    """Same operations, independent weights per call, identical initial function."""
    def __init__(self, config, *, patch_count=9):
        super().__init__(config, patch_count=patch_count)
        template = self.reasoner
        self.reasoner = nn.ModuleList([
            nn.ModuleList([copy.deepcopy(template)
                           for _ in range(config["memory_updates"] + 1)])
            for _ in range(config["refinement_steps"])])

    def refine_tokens(self, image, answer, memory, step):
        stacks = self.reasoner[step]
        for stack in stacks[:-1]:
            memory = stack(torch.cat((image, answer, memory), dim=1))[:, -memory.shape[1]:]
        answer = stacks[-1](torch.cat((answer, memory), dim=1))[:, :1]
        return answer, memory


def make_depth_model(variant, seed, model_config=None, *, patch_count=9):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown depth comparison: {variant}")
    config = dict(model_config or json.loads((PROJECT / "configs/vitrm.json").read_text())["model"])
    config["refinement_steps"] = 1 if variant == "shared_1" else 4
    seed_everything(seed)
    cls = UntiedVisionDepth if variant == "untied_4" else VisionViTRM
    return cls(config, patch_count=patch_count)


def prefix_reports(logits, labels, frame, depths=DEPTHS):
    """Paired metrics; corrections/regressions use the same image and checkpoint."""
    if logits.ndim != 3:
        raise ValueError("Expected finite [images,depth,4] logits")
    depths = tuple(depths)
    if (not depths or tuple(sorted(set(depths))) != depths or
            any(type(depth) is not int or not 1 <= depth <= logits.shape[1] for depth in depths)):
        raise ValueError("Requested depths must be distinct, increasing trained prefixes")
    if (logits.ndim != 3 or logits.shape[0] != len(labels) or logits.shape[2] != 4 or
            len(frame) != len(labels) or len(labels) == 0 or not torch.isfinite(logits).all()):
        raise ValueError("Expected finite [images,depth,4] logits and aligned nonempty labels")
    if labels.ndim != 1 or labels.dtype != torch.long or ((labels < 0) | (labels > 3)).any():
        raise ValueError("Expected integer severity labels 0–3")
    if frame.file_path.duplicated().any() or frame.severity.astype(int).tolist() != labels.tolist():
        raise ValueError("Image identifiers and labels must align without duplicate images")
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()
    probabilities = logits.softmax(-1)
    losses = -logits.log_softmax(-1).gather(-1, labels[:, None, None].expand(-1, logits.shape[1], 1)).squeeze(-1)
    predictions = logits.argmax(-1)
    correct = predictions == labels[:, None]
    routes, true_routes = predictions >= 2, labels >= 2
    route_correct = routes == true_routes[:, None]
    abs_error = (predictions - labels[:, None]).abs()
    per_image, summaries, transitions, pairs = [], [], [], []
    for depth in depths:
        index = depth - 1
        summaries.append({"step": depth, "images": len(labels),
            "severity_accuracy": correct[:, index].float().mean().item(),
            "cross_entropy": losses[:, index].mean().item(),
            "severity_mae": abs_error[:, index].float().mean().item(),
            "routing_errors": (~route_correct[:, index]).sum().item(),
            "route_false_positives": (routes[:, index] & ~true_routes).sum().item(),
            "route_false_negatives": (~routes[:, index] & true_routes).sum().item()})
        for row_index, row in enumerate(frame.itertuples(index=False)):
            per_image.append({"image_path": str(row.file_path), "subject_id": str(row.subject_id),
                "step": depth, "true_severity": labels[row_index].item(),
                "predicted_severity": predictions[row_index, index].item(),
                "true_route": int(true_routes[row_index]), "predicted_route": int(routes[row_index, index]),
                "severity_correct": bool(correct[row_index, index]),
                "cross_entropy": losses[row_index, index].item(),
                "severity_absolute_error": abs_error[row_index, index].item(),
                "routing_error": not bool(route_correct[row_index, index]),
                **{f"probability_{c}": probabilities[row_index, index, c].item() for c in range(4)}})
    for early, late in combinations(depths, 2):
        a, b = early - 1, late - 1
        corrected, regressed = ~correct[:, a] & correct[:, b], correct[:, a] & ~correct[:, b]
        route_corrected = ~route_correct[:, a] & route_correct[:, b]
        route_regressed = route_correct[:, a] & ~route_correct[:, b]
        pair = {"from_step": early, "to_step": late, "images": len(labels),
            "corrections": corrected.sum().item(), "regressions": regressed.sum().item(),
            "net_corrections": (corrected.sum() - regressed.sum()).item(),
            "unchanged_correct": (correct[:, a] & correct[:, b]).sum().item(),
            "unchanged_incorrect": (~correct[:, a] & ~correct[:, b]).sum().item(),
            "routing_corrections": route_corrected.sum().item(),
            "routing_regressions": route_regressed.sum().item(),
            "cross_entropy_delta": (losses[:, b] - losses[:, a]).mean().item(),
            "severity_mae_delta": (abs_error[:, b] - abs_error[:, a]).float().mean().item(),
            "routing_error_delta": ((~route_correct[:, b]).sum() - (~route_correct[:, a]).sum()).item()}
        pairs.append(pair)
        for index, row in enumerate(frame.itertuples(index=False)):
            transitions.append({"image_path": str(row.file_path), "subject_id": str(row.subject_id),
                "true_severity": labels[index].item(), "from_step": early, "to_step": late,
                "earlier_prediction": predictions[index, a].item(), "later_prediction": predictions[index, b].item(),
                "corrected": bool(corrected[index]), "regressed": bool(regressed[index]),
                "routing_corrected": bool(route_corrected[index]), "routing_regressed": bool(route_regressed[index]),
                "cross_entropy_delta": (losses[index, b] - losses[index, a]).item(),
                "severity_absolute_error_delta": (abs_error[index, b] - abs_error[index, a]).item()})
    return {"prefix_predictions": pd.DataFrame(per_image), "prefix_metrics": pd.DataFrame(summaries),
            "image_transitions": pd.DataFrame(transitions), "transition_metrics": pd.DataFrame(pairs)}


def collect_logits(model, data, batch_size):
    """Exactly one encoder call per batch, then one complete trained unroll."""
    model.eval()
    outputs = []
    with torch.inference_mode():
        for pixels, _ in DataLoader(data, batch_size=batch_size):
            image = model.encode(pixels)
            outputs.append(model.forward_tokens(image))
    return torch.cat(outputs)


def benchmark_prefixes(model, pixels, depths, repeats=5):
    """Warmed CPU model timing; exclude encoding to isolate depth cost."""
    model.eval()
    rows = []
    with torch.inference_mode():
        image = model.encode(pixels)
        for depth in depths:
            for _ in range(2):
                model.forward_tokens(image, steps=depth)
            samples = []
            for _ in range(repeats):
                start = perf_counter()
                model.forward_tokens(image, steps=depth)
                samples.append(perf_counter() - start)
            rows.append({"step": depth, "batch_size": len(pixels), "repeats": repeats,
                "median_batch_seconds_excluding_encoder": median(samples),
                "median_seconds_per_image_excluding_encoder": median(samples) / len(pixels)})
    return pd.DataFrame(rows)


def evaluate_selected(output, protocol, split, frame, rows, *, test_data=None):
    """Called only after ALL checkpoint selection is complete; never retrains."""
    root = Path(protocol["dataset_root"])
    validate_subject_split(split, frame, root)
    test_frame = partition_frame(split, "test", root)
    print(f"All checkpoints fixed; loading {len(test_frame)} test images", flush=True)
    test = preload(test_frame, protocol["preprocessing"]) if test_data is None else test_data
    if len(test) != len(test_frame) or test.tensors[1].tolist() != test_frame.severity.astype(int).tolist():
        raise ValueError("Cached test data must align with the saved test manifest")
    tables = {name: [] for name in ("prefix_predictions", "prefix_metrics", "image_transitions", "transition_metrics", "prefix_timing")}
    factory = partial(make_depth_model, model_config=protocol["model_config"],
                      patch_count=protocol["preprocessing"]["grid_size"] ** 2)
    for row in rows:
        directory = output / f"{row['variant']}_seed{row['seed']}"
        checkpoint = torch.load(directory / "best.pt", weights_only=True)
        if (checkpoint["schema_version"] != SCHEMA or checkpoint["protocol"] != protocol or
                checkpoint["variant"] != row["variant"] or checkpoint["seed"] != row["seed"] or
                file_sha256(directory / "best.pt") != row["checkpoint_sha256"]):
            raise ValueError("Checkpoint identity or protocol changed")
        model = factory(row["variant"], row["seed"])
        model.load_state_dict(checkpoint["model_state"])
        logits = collect_logits(model, test, protocol["batch_size"])
        depths = tuple(depth for depth in DEPTHS if depth <= model.config["refinement_steps"])
        reports = prefix_reports(logits, test.tensors[1], test_frame, depths)
        reports["prefix_timing"] = benchmark_prefixes(model, test.tensors[0][:protocol["batch_size"]], depths)
        for name, table in reports.items():
            table = table.assign(variant=row["variant"], seed=row["seed"],
                                 checkpoint_sha256=row["checkpoint_sha256"])
            if name == "prefix_metrics":
                table = table.assign(parameters=row["parameters"], selected_epoch=row["selected_epoch"])
            table.to_csv(directory / f"{name}.csv", index=False)
            tables[name].append(table)
        print(f"TEST {row['variant']} seed={row['seed']}: "
              f"{reports['prefix_metrics'][['step', 'severity_accuracy', 'cross_entropy']].to_dict('records')}", flush=True)
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_csv(output / f"{name}.csv", index=False)
    metrics = pd.concat(tables["prefix_metrics"], ignore_index=True)
    summary = metrics.groupby(["variant", "step"], sort=False).agg(
        runs=("seed", "count"), accuracy_mean=("severity_accuracy", "mean"),
        accuracy_std=("severity_accuracy", "std"), cross_entropy_mean=("cross_entropy", "mean"),
        severity_mae_mean=("severity_mae", "mean"), routing_errors_mean=("routing_errors", "mean"),
        parameters=("parameters", "first"))
    summary.to_csv(output / "comparison_summary.csv")
    atomic_json(output / "evaluation.json", {"test_images": len(test_frame),
        "test_subjects": int(test_frame.subject_id.nunique()), "split_digest": manifest_digest(split),
        "depths": list(DEPTHS), "all_checkpoints_selected_before_test_loading": True,
        "same_checkpoint_and_encoded_tokens_across_prefixes": True,
        "test_status": protocol["test_status"]})
    print(summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--reference-split", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--threads", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validation-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true",
                      help="Evaluate saved checkpoints in an existing validation-only run")
    args = parser.parse_args()
    if min(args.epochs, args.threads) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive epochs/threads and distinct seeds required")
    if not args.evaluate_only and (args.dataset_root is None or args.reference_split is None):
        parser.error("Training requires --dataset-root and --reference-split")
    if args.evaluate_only and (args.dataset_root is not None or args.reference_split is not None):
        parser.error("Evaluation uses the saved dataset and split; do not override them")
    output = args.output.resolve()
    started = perf_counter()
    if args.evaluate_only:
        protocol = json.loads((output / "protocol.json").read_text())
        status = json.loads((output / "run.json").read_text())
        if status["status"] != "ready_for_evaluation":
            parser.error("Evaluation requires a completed validation-only run")
        split = json.loads((output / "split.json").read_text())
        rows = json.loads((output / "selected_checkpoints.json").read_text())
        torch.set_num_threads(protocol["threads"])
    else:
        output.mkdir(parents=True, exist_ok=False)
        config = json.loads((PROJECT / "configs/vitrm.json").read_text())
        protocol = {"checkpoint_schema": SCHEMA, "variants": list(VARIANTS), "seeds": args.seeds,
            "epochs": args.epochs, "batch_size": 8, "learning_rate": 0.0003, "weight_decay": 0.0001,
            "gradient_clip": 1.0, "precision": "float32", "device": "cpu", "threads": args.threads,
            "model_config": {**config["model"], "refinement_steps": 4},
            "preprocessing": config["preprocessing"], "reported_depths": list(DEPTHS),
            "training": "Mean CE equally supervises EVERY output (1,2,3,4); full unroll gradients; one update per batch",
            "selection": "Lowest final-step validation CE, one checkpoint per variant/seed; same checkpoint at all prefixes",
            "controls": "shared_1: separate one-step training. untied_4: independent stack at every memory/answer call, same operations and supervision; more parameters, matched Transformer-call structure",
            "initialization": "All common weights identical within seed. Every untied stack is a deep copy of the shared initial stack; then each model trains independently end to end",
            "test_status": "Exploratory reuse of the previously inspected cohort; not independent confirmation",
            "timing": "CPU FP32, two warmups/five repeats per prefix on one batch; excludes encoder and loading; no FLOP or GPU throughput claim",
            "screening_only": True, "dataset_root": str(args.dataset_root.resolve()),
            "reference_split": str(args.reference_split.resolve()),
            "reference_split_sha256": file_sha256(args.reference_split),
            "started_at": datetime.now(timezone.utc).isoformat()}
        torch.set_num_threads(args.threads)
        atomic_json(output / "protocol.json", protocol)
        status = {"status": "preparing", "completed_training_runs": 0,
                  "planned_runs": len(VARIANTS) * len(args.seeds)}
        atomic_json(output / "run.json", status)
        atomic_json(output / "source_hashes.json", snapshot_source(output))
        atomic_json(output / "environment.json", environment_info(torch.device("cpu"), "float32"))
        split = json.loads(args.reference_split.read_text())
        atomic_json(output / "split.json", split)
        rows = []
    try:
        root = Path(protocol["dataset_root"])
        frame = build_dataset(root)
        validate_subject_split(split, frame, root)
        if not args.evaluate_only:
            train_frame, val_frame = (partition_frame(split, name, root) for name in ("train", "validation"))
            print("Loading training/validation pixels only", flush=True)
            train, validation = preload(train_frame, protocol["preprocessing"]), preload(val_frame, protocol["preprocessing"])
            factory = partial(make_depth_model, model_config=protocol["model_config"],
                              patch_count=protocol["preprocessing"]["grid_size"] ** 2)
            digests = {}
            for seed in protocol["seeds"]:
                for variant in VARIANTS:
                    status.update(status="training", current_variant=variant, current_seed=seed)
                    atomic_json(output / "run.json", status)
                    row = train_one(variant, seed, train, validation,
                                    output / f"{variant}_seed{seed}", protocol, model_factory=factory)
                    if seed in digests and row["initial_feature_digest"] != digests[seed]:
                        raise AssertionError("Initial encoder and positions differ across controls")
                    digests[seed] = row["initial_feature_digest"]
                    rows.append(row)
                    atomic_json(output / "selected_checkpoints.json", rows)
                    status["completed_training_runs"] += 1
                    atomic_json(output / "run.json", status)
            status.update(status="ready_for_evaluation", initial_features_identical_within_seed=True)
            atomic_json(output / "run.json", status)
            del train, validation
        expected = {(variant, seed) for variant in protocol["variants"] for seed in protocol["seeds"]}
        actual = {(row["variant"], row["seed"]) for row in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError("All predefined checkpoints must be selected before evaluation")
        if not args.validation_only:
            status["status"] = "evaluating"
            atomic_json(output / "run.json", status)
            evaluate_selected(output, protocol, split, frame, rows)
            status["status"] = "completed"
        status["elapsed_seconds"] = perf_counter() - started
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(output / "run.json", status)


if __name__ == "__main__":
    main()
