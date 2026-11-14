"""Measure Qwen few-shot inference on one or two CUDA GPUs.

Run ``prepare`` once, then run each ``sample`` in a fresh process so previous
loads and CUDA allocations cannot contaminate the measurements. Model sharding
uses Accelerate; no quantization, CPU offload, or custom image resize is used.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from time import perf_counter
import traceback
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
MODEL = "Qwen/Qwen3.5-0.8B"
DATASET = "trainingdatapro/human-faces-augmentation-image-restoration"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare(args):
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import snapshot_download
    import kagglehub
    from PIL import Image
    from src.dataset_setup import find_dataset_root
    from src.image_quality_pipeline import build_dataset

    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "prepare_status.json", {"status": "preparing"})
    with ThreadPoolExecutor(max_workers=2) as executor:
        model_future = executor.submit(snapshot_download, repo_id=MODEL,
            revision=args.revision, cache_dir=str(args.cache))
        dataset_future = executor.submit(kagglehub.dataset_download, args.dataset)
        snapshot = Path(model_future.result())
        downloaded = Path(dataset_future.result())
    root = find_dataset_root(downloaded)
    frame = build_dataset(root)
    records = []
    for row in frame.itertuples():
        path = Path(row.file_path)
        with Image.open(path) as image:
            width, height = image.size
        records.append({"path": path.relative_to(root).as_posix(),
            "subject": row.subject_id, "severity": row.severity,
            "width": width, "height": height, "pixels": width * height,
            "bytes": path.stat().st_size})
    targets = [row for row in records if row["subject"] != "0"]
    metadata = {"status": "prepared", "model": MODEL, "revision": snapshot.name,
        "snapshot": str(snapshot), "dataset_handle": args.dataset,
        "dataset_root": str(root), "images": records,
        "targets": {"smallest": min(targets, key=lambda row: (row["pixels"], row["bytes"]))["path"],
                    "largest": max(targets, key=lambda row: (row["pixels"], row["bytes"]))["path"]}}
    save(args.output / "prepared.json", metadata)
    save(args.output / "prepare_status.json", {"status": "completed", "targets": metadata["targets"],
                                               "revision": metadata["revision"], "images": len(records)})
    print(json.dumps({key: value for key, value in metadata.items() if key != "images"}), flush=True)


class GPUMonitor:
    """Track driver memory, including CUDA allocations outside PyTorch."""
    def __init__(self):
        self.stop = threading.Event()
        self.peaks = {}
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                output = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                for line in output.stdout.splitlines():
                    index, used, utilization = [int(field.strip()) for field in line.split(",")]
                    previous = self.peaks.setdefault(str(index), {"driver_peak_mib": 0, "max_utilization_percent": 0})
                    previous["driver_peak_mib"] = max(previous["driver_peak_mib"], used)
                    previous["max_utilization_percent"] = max(previous["max_utilization_percent"], utilization)
            except Exception:
                pass
            self.stop.wait(0.5)


def validate_efficient_attention(dtype=None):
    """Check explicit KV expansion against grouped-query float32 math."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    dtype = dtype or torch.float32
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    query = torch.randn(1, 8, 97, 256, device="cuda:0", dtype=dtype, generator=generator)
    key = torch.randn(1, 2, 97, 256, device="cuda:0", dtype=dtype, generator=generator)
    value = torch.randn(1, 2, 97, 256, device="cuda:0", dtype=dtype, generator=generator)
    rtol, atol = (1e-2, 2e-3) if dtype == torch.float16 else (1e-4, 1e-5)
    checks = []
    with torch.inference_mode():
        for causal in (False, True):
            with sdpa_kernel(SDPBackend.MATH):
                reference = scaled_dot_product_attention(query.float(), key.float(), value.float(),
                    is_causal=causal, enable_gqa=True)
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                actual = scaled_dot_product_attention(query, key.repeat_interleave(4, dim=1),
                    value.repeat_interleave(4, dim=1), is_causal=causal)
            torch.testing.assert_close(actual.float(), reference, rtol=rtol, atol=atol)
            checks.append({"causal": causal, "dtype": str(dtype),
                           "max_absolute_error": (actual.float() - reference).abs().max().item(),
                           "rtol": rtol, "atol": atol})
    return checks


def sample(args):
    import torch
    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from src.image_quality_pipeline import (build_dataset, prepare_few_shot_split,
                                            FEW_SHOT_GUIDANCE, parse_prediction)
    from src.qwen_adapter import QwenImageModel

    prepared = json.loads((args.output / "prepared.json").read_text())
    root, snapshot = Path(prepared["dataset_root"]), Path(prepared["snapshot"])
    relative = prepared["targets"].get(args.target, args.target)
    target = root / relative
    label = args.label or (args.placement + "_" + args.target.replace("/", "_"))
    report_path = args.output / (label + ".json")
    if report_path.exists():
        raise FileExistsError(f"Choose a fresh label: {report_path}")
    report = {"status": "loading_model", "started_at": datetime.now(timezone.utc).isoformat(),
        "placement": args.placement, "attention": args.attention, "target": relative, "model": MODEL,
        "allocator": os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
        "split_at": args.split_at if args.placement == "split_layers" else None,
        "revision": prepared["revision"], "dtype": args.dtype, "max_new_tokens": 128,
        "do_sample": False, "custom_pixel_cap": None,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "transformers", "accelerate", "Pillow", "pandas")},
        "gpus": [{"name": torch.cuda.get_device_name(i),
                  "total_mib": torch.cuda.get_device_properties(i).total_memory / 1024**2}
                 for i in range(torch.cuda.device_count())]}
    save(report_path, report)
    monitor = GPUMonitor()
    monitor.thread.start()
    started = perf_counter()
    try:
        dtype = getattr(torch, args.dtype)
        if args.attention == "efficient":
            report["attention_validation"] = validate_efficient_attention(dtype)
        if args.placement != "single" and torch.cuda.device_count() < 2:
            raise RuntimeError("Two CUDA GPUs are required for sharded placement")
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        kwargs = {"dtype": dtype, "local_files_only": True}
        if args.placement == "balanced":
            kwargs.update(device_map="balanced", max_memory={0: "12GiB", 1: "12GiB", "cpu": 0})
        elif args.placement == "vision_text":
            kwargs["device_map"] = {"model.visual": 0, "model.language_model": 1, "lm_head": 1}
        elif args.placement == "split_layers":
            kwargs["device_map"] = {
                "model.visual": 0, "model.language_model.embed_tokens": 0, "lm_head": 0,
                "model.language_model.norm": 1, "model.language_model.rotary_emb": 1,
                **{f"model.language_model.layers.{index}": 0 if index < args.split_at else 1
                   for index in range(24)},
            }
        else:
            kwargs["device_map"] = {"": 0}
        model = AutoModelForMultimodalLM.from_pretrained(snapshot, **kwargs)
        report["allocated_after_model_load_mib"] = torch.cuda.memory_allocated(0) / 1024**2
        report["logit_checks"] = {"calls": 0, "nonfinite_values": 0}
        def check_logits(_module, _arguments, output):
            count = int((~torch.isfinite(output)).sum().item())
            report["logit_checks"]["calls"] += 1
            report["logit_checks"]["nonfinite_values"] += count
            if count:
                raise FloatingPointError(f"Model produced {count} nonfinite raw logits")
        model.lm_head.register_forward_hook(check_logits)
        report["model_load_seconds"] = perf_counter() - started
        report["device_map"] = {name: str(device) for name, device in
                                getattr(model, "hf_device_map", {"": model.device}).items()}
        used = {str(parameter.device) for parameter in model.parameters()}
        report["parameter_devices"] = sorted(used)
        report["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        if any(not device.startswith("cuda:") for device in used):
            raise RuntimeError(f"CPU/disk offload is not allowed in this probe: {used}")
        if args.placement != "single" and used != {"cuda:0", "cuda:1"}:
            raise RuntimeError(f"Sharded placement did not actually use both GPUs: {used}")
        adapter = QwenImageModel(model, processor, model_name=MODEL, snapshot=snapshot,
                                 mode="few_shot", example_subject="0")
        _, examples = prepare_few_shot_split(build_dataset(root), "0")
        report["images"] = []
        for path in [Path(example.image_path) for example in examples] + [target]:
            with Image.open(path) as image:
                size = list(image.size)
            report["images"].append({"path": path.relative_to(root).as_posix(), "size": size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        report["processor_size"] = dict(processor.image_processor.size)
        report["processor_do_resize"] = processor.image_processor.do_resize
        original_template = processor.apply_chat_template
        def capture_inputs(*positional, **named):
            inputs = original_template(*positional, **named)
            report["input_shapes"] = {key: list(value.shape) for key, value in inputs.items()
                                      if hasattr(value, "shape")}
            if "image_grid_thw" in inputs:
                report["image_grid_thw"] = inputs["image_grid_thw"].tolist()
            report["status"] = "generating"
            save(report_path, report)
            return inputs
        processor.apply_chat_template = capture_inputs
        prompt = (PROJECT / "configs/qwen_original_prompt.txt").read_text() + FEW_SHOT_GUIDANCE
        report["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
            torch.cuda.reset_peak_memory_stats(index)
        report["status"] = "processing_images"
        save(report_path, report)
        generation_started = perf_counter()
        with ExitStack() as attention_context:
            if args.attention == "efficient":
                from torch.nn.attention import SDPBackend, sdpa_kernel
                # CUDA grouped-query SDPA can fall back to the quadratic math
                # kernel in float32. The existing Transformers implementation
                # expands KV heads when this helper returns False. Force the
                # efficient kernel so unsupported dispatch fails visibly rather
                # than silently allocating a full attention matrix.
                attention_context.enter_context(patch(
                    "transformers.integrations.sdpa_attention.use_gqa_in_sdpa",
                    new=lambda *_arguments, **_keywords: False))
                attention_context.enter_context(sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION))
            raw = adapter.generate(target, prompt, examples=examples)
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
        report["generation_seconds"] = perf_counter() - generation_started
        report["allocated_after_call_mib"] = torch.cuda.memory_allocated(0) / 1024**2
        report["raw_output"] = raw
        try:
            report["prediction"] = parse_prediction(raw)
            report["valid_json_prediction"] = True
        except ValueError as error:
            report["valid_json_prediction"] = False
            report["prediction_error"] = str(error)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = perf_counter() - started
        report["torch_memory"] = {}
        for index in range(torch.cuda.device_count()):
            report["torch_memory"][str(index)] = {
                "peak_allocated_mib": torch.cuda.max_memory_allocated(index) / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(index) / 1024**2}
        monitor.stop.set()
        monitor.thread.join(timeout=6)
        report["driver_memory"] = monitor.peaks
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(report_path, report)
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "completed" else 1


def suite(args):
    """Run serially in fresh processes; bound each attempt to 15 minutes."""
    if not (args.output / "prepared.json").is_file():
        raise FileNotFoundError("Run prepare before the inference suite")
    summary = {"status": "running", "attempts": []}
    for placement, target in (("balanced", "smallest"), ("balanced", "largest"),
                              ("single", "smallest"), ("single", "largest")):
        label = ("efficient_" if args.attention == "efficient" else "") + placement + "_" + target
        report_path = args.output / (label + ".json")
        if report_path.exists():
            raise FileExistsError(f"Use a fresh output directory: {report_path}")
        summary["current"] = label
        save(args.output / "suite_status.json", summary)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "sample",
                   "--output", str(args.output), "--placement", placement, "--target", target,
                   "--attention", args.attention, "--label", label, "--dtype", args.dtype]
        with (args.output / (label + ".log")).open("w") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                exit_code = "timeout"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        summary["attempts"].append({"label": label, "exit_code": exit_code,
            "status": report.get("status", "failed_before_report"),
            "generation_seconds": report.get("generation_seconds"),
            "torch_memory": report.get("torch_memory"), "error": report.get("error")})
        save(args.output / "suite_status.json", summary)
    summary.update(status="completed", current=None)
    save(args.output / "suite_status.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "sample", "suite"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=PROJECT / "hf_cache/hub")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--placement", choices=["single", "balanced", "vision_text", "split_layers"], default="balanced")
    parser.add_argument("--attention", choices=["auto", "efficient"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--split-at", type=int, choices=range(1, 24), default=12)
    parser.add_argument("--allocator", choices=["default", "expandable"], default="default")
    parser.add_argument("--target", default="smallest")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    if args.allocator == "expandable":
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    return {"prepare": prepare, "sample": sample, "suite": suite}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
