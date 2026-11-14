"""Run the largest-input FP16 stress test and a fixed FP32/FP16 comparison.

Use a PyTorch CUDA runtime with dataset/model Internet access. The declared
manifest fixes the cases before inference; this runner records their hashes
before launching any predictions. Each precision gets a fresh, serial worker
on the same physical GPU. Source and reports are retained with the run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

from probe_qwen_memory import save

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-prepared", type=Path)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "suite_status.json"
    if status_path.exists():
        raise FileExistsError(f"Choose a new run directory: {args.output}")
    declared = json.loads(args.manifest.read_text())
    if len(declared["cases"]) != 12 or declared["subjects"] != ["27", "14", "9"]:
        raise ValueError("Unexpected accuracy subset; review the declaration before running")
    if len({row["path"] for row in declared["cases"]}) != 12:
        raise ValueError("Duplicate target cases")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
        PYTORCH_ALLOC_CONF="expandable_segments:True")
    status = {"status": "preparing", "started_at": datetime.now(timezone.utc).isoformat(),
              "physical_gpu": args.gpu, "concurrent_workers": False, "attempts": []}
    save(status_path, status)
    started = perf_counter()

    def run(stage, command, timeout):
        status.update(status="running", stage=stage)
        save(status_path, status)
        with (args.output / (stage + ".log")).open("w") as log:
            try:
                process = subprocess.run(command, cwd=PROJECT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        status["attempts"].append({"stage": stage, "exit_code": code})
        save(status_path, status)
        return code

    try:
        prepared_path = args.output / "prepared.json"
        reuse = args.reuse_prepared
        reusable = None
        if reuse and reuse.is_file():
            candidate = json.loads(reuse.read_text())
            if (candidate["revision"] == declared["model_revision"]
                    and Path(candidate["snapshot"]).is_dir()
                    and Path(candidate["dataset_root"]).is_dir()):
                reusable = candidate
        if reusable:
            save(prepared_path, reusable)
        else:
            code = run("prepare", [sys.executable, "-u", str(PROJECT / "scripts/probe_qwen_memory.py"),
                "prepare", "--output", str(args.output), "--revision", declared["model_revision"],
                "--cache", str(PROJECT / "hf_cache/hub")], 900)
            if code != 0:
                raise RuntimeError(f"Model/dataset preparation failed: {code}")
        prepared = json.loads(prepared_path.read_text())
        root = Path(prepared["dataset_root"])
        from PIL import Image
        resolved = json.loads(json.dumps(declared))
        for image in resolved["references"] + resolved["cases"]:
            relative = Path(image["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Expected a relative dataset image: {relative}")
            path = root / relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if image.get("sha256") and digest != image["sha256"]:
                raise ValueError(f"Image differs from the declaration: {relative}")
            with Image.open(path) as pixels:
                if "width" in image and pixels.size != (image["width"], image["height"]):
                    raise ValueError(f"Image dimensions changed: {relative}")
            image["sha256"] = digest
        resolved.update(status="frozen_before_inference", frozen_at=datetime.now(timezone.utc).isoformat())
        manifest_path = args.output / "subset_frozen.json"
        save(manifest_path, resolved)
        source_hashes = {}
        for relative in ["scripts/probe_qwen_memory.py", "scripts/evaluate_qwen_precision.py",
                "scripts/run_qwen_precision_evaluation.py", "src/qwen_adapter.py",
                "src/image_quality_pipeline.py", "src/evaluation.py", "configs/qwen_original_prompt.txt"]:
            source = PROJECT / relative
            target = args.output / "tested_source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            source_hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
        save(args.output / "source_sha256.json", source_hashes)

        for precision, label in (("float16", "fp16_largest"), ("float32", "fp32_largest")):
            run(label, [sys.executable, "-u", str(PROJECT / "scripts/probe_qwen_memory.py"),
                "sample", "--output", str(args.output), "--placement", "single", "--attention", "efficient",
                "--dtype", precision, "--allocator", "expandable", "--target", "largest",
                "--label", label], 900)
        for precision in ("float32", "float16"):
            run(precision, [sys.executable, "-u", str(PROJECT / "scripts/evaluate_qwen_precision.py"),
                "--prepared", str(prepared_path), "--manifest", str(manifest_path),
                "--output", str(args.output / precision), "--dtype", precision, "--threads", "2"], 2400)
        status.update(status="completed", stage=None)
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        status.update(elapsed_seconds=perf_counter() - started,
                      finished_at=datetime.now(timezone.utc).isoformat())
        save(status_path, status)
        print(json.dumps(status, indent=2), flush=True)
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
