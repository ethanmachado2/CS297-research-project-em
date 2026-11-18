"""Run one fresh Kaggle workload: python -m src.run --help."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil

PROJECT = Path(__file__).resolve().parents[1]


def select_qwen_targets(frame, config, selection):
    from .image_quality_pipeline import prepare_few_shot_split
    targets, examples = prepare_few_shot_split(frame, config["reference_subject"])
    if selection == "smoke":
        targets = targets.loc[targets.subject_id.isin(config["smoke_subjects"])].copy()
        # Freeze the same subject-major order used in the precision evaluation.
        order = {subject: index for index, subject in enumerate(config["smoke_subjects"])}
        targets = targets.assign(_order=targets.subject_id.map(order)).sort_values(["_order", "severity"])
        targets = targets.drop(columns="_order")
        if len(targets) != 12:
            raise ValueError("Smoke selection requires all 12 predefined images")
    elif selection == "largest":
        suffix = config["largest_target"]
        targets = targets.loc[targets.file_path.map(lambda p: Path(p).as_posix().endswith("/" + suffix))]
        if len(targets) != 1:
            raise ValueError(f"Largest target is missing: {suffix}")
    elif selection != "full":
        raise ValueError(f"Unknown selection: {selection}")
    return targets, examples


def snapshot_source(output):
    hashes = {}
    for folder in ("src", "configs", "vendor"):
        for path in sorted((PROJECT / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = path.relative_to(PROJECT)
            destination = output / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("requirements.txt", "README.md"):
        shutil.copy2(PROJECT / name, output / "source" / name)
        hashes[name] = hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["qwen_single_shot", "qwen_few_shot", "trm_supervised", "vitrm_supervised"])
    parser.add_argument("--selection", choices=["full", "smoke", "largest"], default="full",
                        help="Qwen: 196 / fixed 12 / one largest input; TRM: full or smoke")
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index; exactly one GPU per worker")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dataset-root", type=Path, help="Optional existing Kaggle dataset directory")
    parser.add_argument("--output", type=Path, help="Fresh directory; existing directories are never overwritten")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="Supervised training configuration JSON")
    source.add_argument("--checkpoint", type=Path, help="Evaluate a selected supervised checkpoint without training")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="ViTRM also supports CPU FP32 smoke checks; other workloads require CUDA")
    parser.add_argument("--refinement-steps", type=int, help="ViTRM training depth; use separate validation-selected runs")
    parser.add_argument("--validation-only", action="store_true", help="Train/select a checkpoint without evaluating test pixels")
    args = parser.parse_args()
    supervised = args.task in {"trm_supervised", "vitrm_supervised"}
    if args.gpu < 0 or args.threads < 1:
        parser.error("gpu must be nonnegative and threads must be positive")
    if supervised and args.selection == "largest":
        parser.error("TRM supports full and smoke selections")
    if not supervised and (args.config or args.checkpoint):
        parser.error("--config and --checkpoint apply only to supervised workloads")
    if args.device == "cpu" and args.task != "vitrm_supervised":
        parser.error("CPU execution is supported only for vitrm_supervised")
    if args.refinement_steps is not None and (args.task != "vitrm_supervised" or args.refinement_steps < 1 or args.checkpoint):
        parser.error("--refinement-steps requires ViTRM training and a positive depth")
    if args.checkpoint and args.selection != "full":
        parser.error("Checkpoint-only evaluation uses --selection full and the saved test split")
    if args.validation_only and (not supervised or args.checkpoint):
        parser.error("--validation-only applies only to supervised training")
    # Set before importing any module that imports PyTorch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from .dataset import ensure_dataset
    from .image_quality_pipeline import build_dataset, run_inference, FEW_SHOT_GUIDANCE
    from .trm_training import atomic_json, seed_everything
    from .trm_reporting import export_saved_run

    if args.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Enable a Kaggle GPU accelerator and select a valid --gpu index")
    torch.set_num_threads(args.threads)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output or PROJECT / "outputs" / f"{args.task}_{args.selection}_{timestamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "preparing", "task": args.task, "selection": args.selection,
              "started_at": datetime.now(timezone.utc).isoformat(), "physical_gpu": args.gpu,
              "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "device": args.device, "evaluation_only": args.checkpoint is not None,
              "python": platform.python_version(),
              "allocator": os.environ["PYTORCH_ALLOC_CONF"], "threads": args.threads,
              "packages": {}}
    for name in ("torch", "torchvision", "transformers", "accelerate", "numpy", "pandas",
                 "Pillow", "einops", "pydantic", "kagglehub"):
        try:
            status["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            status["packages"][name] = None
    adapter = None
    try:
        status["source_sha256"] = snapshot_source(output)
        atomic_json(output / "run.json", status)
        root, dataset_manifest, source = ensure_dataset(PROJECT, dataset_root=args.dataset_root)
        atomic_json(output / "dataset.json", {"source": source, **dataset_manifest})
        dataset = build_dataset(root)
        status["status"] = "running"
        atomic_json(output / "run.json", status)
        if supervised:
            from .trm_training import architecture, load_checkpoint, train_experiment, validate_config
            from .trm_adapter import evaluate_checkpoint
            expected = "vitrm" if args.task == "vitrm_supervised" else "trm"
            if args.checkpoint:
                checkpoint = args.checkpoint.resolve()
                _, saved = load_checkpoint(checkpoint, "cpu")
                config = saved["config"]
            else:
                config = json.loads((args.config or PROJECT / f"configs/{expected}.json").read_text())
                config["training"]["device"] = args.device
                if args.device == "cpu":
                    config["training"]["precision"] = "float32"
                if args.refinement_steps is not None:
                    config["model"]["refinement_steps"] = args.refinement_steps
                if args.selection == "smoke":
                    config["training"]["max_epochs"] = 1
            if architecture(config) != expected:
                raise ValueError("Selected workload does not match configuration/checkpoint architecture")
            validate_config(config)
            if not args.checkpoint:
                checkpoint = train_experiment(dataset, root, output, config)
            if args.validation_only:
                status.update(status="completed", checkpoint=str(checkpoint), test_evaluated=False)
                print(f"Validation-selected checkpoint: {checkpoint}", flush=True)
                return
            run = evaluate_checkpoint(checkpoint, dataset, root, output, device=args.device)
        else:
            from .qwen_adapter import QwenImageModel
            config = json.loads((PROJECT / "configs/qwen.json").read_text())
            seed_everything(config["seed"])
            targets, examples = select_qwen_targets(dataset, config, args.selection)
            mode = "few_shot" if args.task == "qwen_few_shot" else "zero_shot"
            prompt = (PROJECT / "configs/qwen_prompt.txt").read_text()
            effective_prompt = prompt + (FEW_SHOT_GUIDANCE if mode == "few_shot" else "")
            atomic_json(output / "selection.json", {"config": config, "mode": mode,
                "reference_subject_excluded_in_both_modes": config["reference_subject"],
                "prompt_sha256": hashlib.sha256(effective_prompt.encode()).hexdigest(),
                "target_paths": [Path(p).relative_to(root).as_posix() for p in targets.file_path],
                "reference_paths": [Path(e.image_path).relative_to(root).as_posix() for e in examples]
                    if mode == "few_shot" else []})
            adapter = QwenImageModel.from_pretrained(config["model"], revision=config["revision"],
                cache_dir=os.environ.get("HF_HUB_CACHE"), mode=mode,
                example_subject=config["reference_subject"] if mode == "few_shot" else None)
            run_inference(adapter, targets.file_path, prompt, labels=targets, output_dir=output,
                experiment_number=2, run_number=1, model_name=config["model"], inference_mode=mode,
                examples=examples if mode == "few_shot" else (), batch_size=1)
            run = output / "experiment_002/run_001"
        report = export_saved_run(run)
        status.update(status="completed" if report["failures"].empty else "completed_with_errors",
                      results_directory=str(run), planned=report["planned"], attempted=report["attempted"],
                      failed=len(report["failures"]))
        print(report["summary"].to_string(), flush=True)
        print(f"Results: {run}", flush=True)
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if adapter is not None:
            atomic_json(output / "gpu_telemetry.json", adapter.telemetry)
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "run.json", status)


if __name__ == "__main__":
    main()
