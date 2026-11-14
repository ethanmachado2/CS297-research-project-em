"""Device-aware, step-supervised training and portable validation checkpoints.

Invoked by python -m src.run trm_supervised
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .trm_data import (FaceSeverityDataset, create_subject_split, file_sha256,
                       manifest_digest, partition_frame, validate_image_sizes,
                       validate_preprocessing)
from .trm_model import UPSTREAM_SHA, VisionTRM, validate_model_config


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def select_device(requested="auto"):
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if requested == "auto":
        requested = next(name for name in ("cuda", "mps", "cpu") if available[name])
    if requested not in available or not available[requested]:
        raise ValueError(f"Requested device {requested!r} is unavailable; available: {available}")
    return torch.device(requested)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported experiment schema version")
    validate_preprocessing(config["preprocessing"])
    validate_model_config(config["model"])
    for value in (config["seed"], config["split"]["seed"]):
        if type(value) is not int or not 0 <= value < 2 ** 32:
            raise ValueError("Seeds must be nonnegative 32-bit integers")
    train = config["training"]
    if train.get("precision") != "float16":
        raise ValueError("Experiment 2 requires FP16 AMP")
    for key in ("batch_size", "max_epochs", "patience"):
        if type(train[key]) is not int or train[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(train["num_workers"]) is not int or train["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    for key in ("learning_rate", "weight_decay", "gradient_clip"):
        if not math.isfinite(train[key]) or train[key] < 0 or (key != "weight_decay" and train[key] == 0):
            raise ValueError(f"Invalid {key}")


def environment_info(device):
    return {"python": platform.python_version(), "platform": platform.platform(), "device": str(device),
            "dtype": "float16_autocast", "parameter_dtype": "float32", "libraries": {name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "pandas", "Pillow", "pydantic", "einops")}}


def validation_metrics(model, loader, device):
    model.eval()
    total_loss = total = correct = 0
    matrix = torch.zeros(4, 4, dtype=torch.int64)
    with torch.inference_mode():
        for patches, targets in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(patches.to(device))
            targets = targets.to(device)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite validation logits")
            loss = nn.functional.cross_entropy(logits.float(), targets, reduction="sum")
            total_loss += loss.item()
            guesses = logits.argmax(-1)
            total += targets.numel()
            correct += (guesses == targets).sum().item()
            for truth, guess in zip(targets.cpu().tolist(), guesses.cpu().tolist()):
                matrix[truth, guess] += 1
    if not total or not math.isfinite(total_loss):
        raise FloatingPointError("Empty or non-finite validation result")
    f1 = []
    for i in range(4):
        denominator = matrix[i].sum().item() + matrix[:, i].sum().item()
        f1.append(2 * matrix[i, i].item() / denominator if denominator else 0.0)
    return {"loss": total_loss / total, "accuracy": correct / total, "macro_f1": sum(f1) / 4, "images": total}


def load_checkpoint(path, device="cpu"):
    """Reconstruct from local weights only; never deserialize arbitrary classes."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema version")
    validate_config(checkpoint["config"])
    if checkpoint["upstream_sha"] != UPSTREAM_SHA:
        raise ValueError("Checkpoint upstream source does not match installed TRM")
    if checkpoint["split"].get("schema_version") != 1 or manifest_digest(checkpoint["split"]) != checkpoint["split_digest"]:
        raise ValueError("Checkpoint split schema/digest mismatch")
    chosen_device = select_device(device)
    config = checkpoint["config"]
    model = VisionTRM(config["model"], patch_count=config["preprocessing"]["grid_size"] ** 2)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(chosen_device).eval()
    return model, checkpoint


def train_experiment(dataset, project_root, session_dir, config, *, show_progress=True):
    """Train on train/validation only. Returns best path; never accesses test pixels."""
    session = Path(session_dir)
    session.mkdir(parents=True, exist_ok=True)
    manifest_path = session / "training_manifest.json"
    if manifest_path.exists() or (session / "checkpoints/best.pt").exists():
        raise FileExistsError(f"Training session already exists: {session}")
    started = perf_counter()
    manifest = {"schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                "config": copy.deepcopy(config), "upstream_sha": UPSTREAM_SHA, "optimizer_updates": 0}
    atomic_json(manifest_path, manifest)
    best_path = session / "checkpoints/best.pt"
    history, best_loss, stale, updates = [], float("inf"), 0, 0
    try:
        validate_config(config)
        device = select_device(config["training"]["device"])
        if device.type != "cuda":
            raise RuntimeError("Experiment 2 FP16 training requires CUDA")
        seed_everything(config["seed"])
        atomic_json(session / "config.json", config)
        split = create_subject_split(dataset, config["split"]["seed"], project_root=project_root,
                                     counts=config["split"]["counts"])
        # Header checks cover every real image before training; only train/val pixels feed learning.
        validate_image_sizes(dataset, config["preprocessing"])
        atomic_json(session / "split.json", split)
        train_frame = partition_frame(split, "train", project_root)
        val_frame = partition_frame(split, "validation", project_root)
        opts = config["training"]
        train_loader = DataLoader(FaceSeverityDataset(train_frame, config["preprocessing"]),
            batch_size=opts["batch_size"], shuffle=True, drop_last=False, num_workers=opts["num_workers"],
            generator=torch.Generator().manual_seed(config["seed"]))
        val_loader = DataLoader(FaceSeverityDataset(val_frame, config["preprocessing"]),
            batch_size=opts["batch_size"], shuffle=False, num_workers=opts["num_workers"])
        model = VisionTRM(config["model"], patch_count=config["preprocessing"]["grid_size"] ** 2).to(device)
        scaler = torch.amp.GradScaler("cuda")
        skipped_updates = 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=opts["learning_rate"], weight_decay=opts["weight_decay"])
        manifest.update(environment=environment_info(device), parameter_counts=model.parameter_counts(),
                        train_images=len(train_frame), validation_images=len(val_frame), split_digest=manifest_digest(split))
        atomic_json(manifest_path, manifest)
        for epoch in range(1, opts["max_epochs"] + 1):
            model.train()
            weighted_loss = step_images = training_images = 0
            epoch_started = perf_counter()
            for patches, targets in train_loader:
                patches, targets = patches.to(device), targets.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    state = model.initial_state(len(targets), device)
                training_images += len(targets)
                for _ in range(config["model"]["refinement_steps"]):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        encoded = model.encode(patches)
                        state, logits = model.refine_once(encoded, state)
                        loss = nn.functional.cross_entropy(logits, targets)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), opts["gradient_clip"])
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped = scaler.get_scale() < previous_scale
                    skipped_updates += int(skipped)
                    updates += int(not skipped)
                    weighted_loss += loss.item() * len(targets)
                    step_images += len(targets)
            if updates == 0:
                raise FloatingPointError("FP16 scaler skipped every optimizer update; no model was trained")
            val = validation_metrics(model, val_loader, device)
            if val["loss"] < best_loss - 1e-6:
                best_loss, stale = val["loss"], 0
                checkpoint = {"schema_version": 1, "model_state": {key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}, "config": copy.deepcopy(config),
                    "split": split, "split_digest": manifest_digest(split), "upstream_sha": UPSTREAM_SHA,
                    "epoch": epoch, "validation_loss": best_loss, "training_status": "running",
                    "optimizer_updates": updates, "training_session": str(session.resolve())}
                atomic_checkpoint(best_path, checkpoint)
                manifest.update(best_epoch=epoch, best_validation_loss=best_loss)
            else:
                stale += 1
            history.append({"epoch": epoch, "training_step_loss": weighted_loss / step_images,
                            "training_images": training_images, "validation_loss": val["loss"],
                            "severity_accuracy": val["accuracy"], "severity_macro_f1": val["macro_f1"],
                            "optimizer_updates": updates, "skipped_updates": skipped_updates, "amp_scale": scaler.get_scale(), "elapsed_seconds": perf_counter() - epoch_started})
            temporary = session / "training_history.csv.tmp"
            pd.DataFrame(history).to_csv(temporary, index=False)
            os.replace(temporary, session / "training_history.csv")
            manifest.update(completed_epochs=epoch, optimizer_updates=updates, skipped_updates=skipped_updates, stale_epochs=stale)
            atomic_json(manifest_path, manifest)
            if show_progress:
                print(f"Epoch {epoch}: train={history[-1]['training_step_loss']:.4f} val={val['loss']:.4f} "
                      f"accuracy={val['accuracy']:.3f} best={manifest['best_epoch']} ({history[-1]['elapsed_seconds']:.1f}s)", flush=True)
            if stale >= opts["patience"]:
                break
        manifest.update(status="completed", stopped_early=stale >= opts["patience"])
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                        error=f"{type(exc).__name__}: {exc}", optimizer_updates=updates)
        raise
    finally:
        manifest.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=perf_counter() - started)
        if best_path.exists():
            saved = torch.load(best_path, map_location="cpu", weights_only=True)
            saved["training_status"] = manifest["status"]
            atomic_checkpoint(best_path, saved)
            manifest["checkpoint_sha256"] = file_sha256(best_path)
        atomic_json(manifest_path, manifest)
    return best_path
