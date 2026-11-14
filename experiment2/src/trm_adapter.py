"""Experiment 2 FP16 AMP adapter: trained TRM logits to the existing strict JSON schema."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .image_quality_pipeline import run_inference
from .trm_data import (file_sha256, partition_frame, load_image_patches,
                       validate_subject_split)
from .trm_training import atomic_json, load_checkpoint


DEFAULT_PROMPT = (Path(__file__).resolve().parents[1] / "configs/qwen_prompt.txt").read_text()


def severity_prediction(severity):
    if severity not in range(4):
        raise ValueError("Severity must be 0–3")
    return {"noise": "0" if severity == 0 else "1", "severity": str(severity),
            "route": "1" if severity >= 2 else "0"}


class TRMImageModel:
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="auto"):
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        return cls(model, checkpoint, checkpoint_path)

    def __init__(self, model, checkpoint, checkpoint_path):
        self.model = model.eval()
        self.checkpoint = checkpoint
        self.preprocessing = checkpoint["config"]["preprocessing"]
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise RuntimeError("Experiment 2 TRM inference requires CUDA FP16 AMP")
        self.inference_config = {
            "uses_prompt": False, "training_regime": "supervised", "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": file_sha256(checkpoint_path), "upstream_sha": checkpoint["upstream_sha"],
            "device": str(self.device), "dtype": "float16_autocast", "parameter_dtype": "float32", "preprocessing": self.preprocessing,
            "model": checkpoint["config"]["model"], "split_digest": checkpoint["split_digest"],
            "parameter_counts": model.parameter_counts(), "refinement_steps": model.config["refinement_steps"],
            "selected_epoch": checkpoint["epoch"], "training_status": checkpoint["training_status"],
            "training_session": checkpoint["training_session"],
        }

    def generate(self, image_path, prompt):
        patches = load_image_patches(image_path, self.preprocessing).unsqueeze(0).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = self.model(patches)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite TRM logits")
        return json.dumps(severity_prediction(int(logits.argmax(-1).item())))


def evaluate_checkpoint(checkpoint_path, dataset, project_root, session_dir, *, device="auto",
                        prompt=DEFAULT_PROMPT, show_progress=True):
    """Validate the checkpoint's original split, then evaluate test images once."""
    adapter = TRMImageModel.from_checkpoint(checkpoint_path, device)
    checkpoint = adapter.checkpoint
    split = validate_subject_split(checkpoint["split"], dataset, project_root)
    session = Path(session_dir)
    session.mkdir(parents=True, exist_ok=True)
    # A new evaluation session records its origin, without pretending to train.
    for name, value in (("split.json", split), ("config.json", checkpoint["config"])):
        destination = session / name
        if destination.exists() and json.loads(destination.read_text()) != value:
            raise ValueError(f"Evaluation session has incompatible {name}")
        atomic_json(destination, value)
    atomic_json(session / "evaluation_origin.json", adapter.inference_config)
    test_frame = partition_frame(split, "test", project_root)
    run_inference(adapter, test_frame.file_path.tolist(), prompt, labels=test_frame, batch_size=1,
                  output_dir=session, experiment_number=2, run_number=1, inference_mode="supervised",
                  response_format="json", model_name="VisionTRM", show_progress=show_progress)
    run = session / "experiment_002/run_001"
    return run
