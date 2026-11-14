"""Evaluate one precision on a predeclared, labeled Qwen few-shot subset.

Run float32 and float16 serially on the same GPU with the same manifest. Images,
prompt, generation settings, and the efficient attention backend stay fixed.
The manifest is selected and saved before any predictions are inspected.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "float16"], required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from probe_qwen_memory import GPUMonitor, save, validate_efficient_attention
    from src.image_quality_pipeline import (build_dataset, prepare_few_shot_split,
        FEW_SHOT_GUIDANCE, parse_prediction, InferenceResult)
    from src.evaluation import evaluate
    from src.qwen_adapter import QwenImageModel

    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    args.output.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    prepared = json.loads(args.prepared.read_text())
    root, snapshot = Path(prepared["dataset_root"]), Path(prepared["snapshot"])
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to a single idle GPU for a matched comparison")
    prompt = (PROJECT / "configs/qwen_original_prompt.txt").read_text() + FEW_SHOT_GUIDANCE
    if hashlib.sha256(prompt.encode()).hexdigest() != manifest["prompt_sha256"]:
        raise ValueError("Prompt differs from the predeclared manifest")
    if snapshot.name != manifest["model_revision"]:
        raise ValueError("Model differs from the predeclared revision")
    for image in manifest["references"] + manifest["cases"]:
        if hashlib.sha256((root / image["path"]).read_bytes()).hexdigest() != image["sha256"]:
            raise ValueError(f"Dataset bytes differ: {image['path']}")
    frame = build_dataset(root)
    _, examples = prepare_few_shot_split(frame, "0")
    report = {"status": "loading_model", "dtype": args.dtype,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "manifest": manifest, "allocator": os.environ["PYTORCH_ALLOC_CONF"],
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2,
        "packages": {name: importlib.metadata.version(name) for name in
            ("torch", "torchvision", "transformers", "accelerate", "Pillow", "pandas")},
        "results": []}
    report_path = args.output / "results.json"
    save(report_path, report)
    monitor = GPUMonitor()
    monitor.thread.start()
    current = {}
    run_started = perf_counter()
    try:
        report["attention_validation"] = validate_efficient_attention(dtype)
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        model_started = perf_counter()
        model = AutoModelForMultimodalLM.from_pretrained(snapshot, local_files_only=True,
            dtype=dtype, device_map={"": 0})
        report["model_load_seconds"] = perf_counter() - model_started
        report["parameter_devices"] = sorted({str(p.device) for p in model.parameters()})
        report["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        if report["parameter_devices"] != ["cuda:0"]:
            raise RuntimeError("Unexpected CPU offload or additional GPU placement")
        adapter = QwenImageModel(model, processor, model_name=manifest["model"],
            snapshot=snapshot, mode="few_shot", example_subject="0")
        report["inference_config"] = adapter.inference_config

        def check_logits(_module, _arguments, output):
            nonfinite = int((~torch.isfinite(output)).sum().item())
            current["logit_forward_calls"] = current.get("logit_forward_calls", 0) + 1
            current["nonfinite_logits"] = current.get("nonfinite_logits", 0) + nonfinite
            if nonfinite:
                raise FloatingPointError(f"Model produced {nonfinite} nonfinite raw logits")
        model.lm_head.register_forward_hook(check_logits)
        original_template = processor.apply_chat_template
        def capture_inputs(*positional, **named):
            started = perf_counter()
            inputs = original_template(*positional, **named)
            current["preprocess_seconds"] = perf_counter() - started
            current["input_shapes"] = {key: list(value.shape) for key, value in inputs.items()
                                       if hasattr(value, "shape")}
            current["image_grid_thw"] = inputs["image_grid_thw"].tolist()
            current["prompt_tokens"] = int(inputs["input_ids"].shape[-1])
            return inputs
        processor.apply_chat_template = capture_inputs
        original_generate = model.generate
        def capture_generation(*positional, **named):
            torch.cuda.synchronize(0)
            started = perf_counter()
            output = original_generate(*positional, **named)
            torch.cuda.synchronize(0)
            current["model_generation_seconds"] = perf_counter() - started
            current["generated_tokens"] = int(output.shape[-1] - named["input_ids"].shape[-1])
            return output
        model.generate = capture_generation

        with ExitStack() as context:
            context.enter_context(patch(
                "transformers.integrations.sdpa_attention.use_gqa_in_sdpa",
                new=lambda *_arguments, **_keywords: False))
            context.enter_context(sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION))
            for index, case in enumerate(manifest["cases"], 1):
                current = {"path": case["path"], "subject_id": case["subject_id"],
                    "truth": case["truth"], "prediction": None, "raw_output": "",
                    "status": "running", "nonfinite_logits": 0}
                report.update(status="running", current_case=case["path"], completed=len(report["results"]))
                save(report_path, report)
                torch.cuda.synchronize(0)
                torch.cuda.reset_peak_memory_stats(0)
                current["allocated_before_mib"] = torch.cuda.memory_allocated(0) / 1024**2
                started = perf_counter()
                try:
                    raw = adapter.generate(root / case["path"], prompt, examples=examples)
                    torch.cuda.synchronize(0)
                    current["raw_output"] = raw
                    current["prediction"] = parse_prediction(raw)
                    current["status"] = "completed"
                except Exception as error:
                    current.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
                        error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                current["elapsed_seconds"] = perf_counter() - started
                current["peak_allocated_mib"] = torch.cuda.max_memory_allocated(0) / 1024**2
                current["peak_reserved_mib"] = torch.cuda.max_memory_reserved(0) / 1024**2
                gc.collect()
                if current["status"] != "completed":
                    torch.cuda.empty_cache()
                current["allocated_after_cleanup_mib"] = torch.cuda.memory_allocated(0) / 1024**2
                report["results"].append(current)
                save(args.output / f"case_{index:02d}.json", current)
                report.update(completed=len(report["results"]), current_case=None)
                save(report_path, report)
                print(json.dumps({key: current.get(key) for key in
                    ("path", "status", "prediction", "elapsed_seconds", "peak_allocated_mib")}), flush=True)
        results = [InferenceResult(row["path"], row["raw_output"], row["prediction"], row.get("error"))
                   for row in report["results"]]
        labels = {case["path"]: case["truth"] for case in manifest["cases"]}
        metrics = evaluate(results, labels)
        for task, values in metrics.items():
            values["end_to_end_accuracy"] = values["correct"] / len(manifest["cases"])
        report.update(status="completed", metrics=metrics)
        save(args.output / "metrics.json", metrics)
        with (args.output / "predictions.csv").open("w", newline="") as handle:
            fields = ["path", "status", "true_severity", "predicted_severity", "true_route",
                "predicted_route", "elapsed_seconds", "peak_allocated_mib", "nonfinite_logits"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in report["results"]:
                prediction = row["prediction"] or {}
                writer.writerow({"path": row["path"], "status": row["status"],
                    "true_severity": row["truth"]["severity"], "predicted_severity": prediction.get("severity"),
                    "true_route": row["truth"]["route"], "predicted_route": prediction.get("route"),
                    "elapsed_seconds": row["elapsed_seconds"], "peak_allocated_mib": row["peak_allocated_mib"],
                    "nonfinite_logits": row["nonfinite_logits"]})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        monitor.stop.set()
        monitor.thread.join(timeout=6)
        report.update(driver_memory=monitor.peaks, elapsed_seconds=perf_counter() - run_started,
                      finished_at=datetime.now(timezone.utc).isoformat())
        save(report_path, report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
