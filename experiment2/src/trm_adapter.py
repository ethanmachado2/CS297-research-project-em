"""Experiment 2 FP16 AMP adapter: trained TRM logits to the existing strict JSON schema."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from .image_quality_pipeline import run_inference
from .trm_data import (file_sha256, partition_frame, load_image_patches,
                       validate_subject_split)
from .trm_training import architecture, atomic_json, autocast_context, load_checkpoint


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
        self.is_vitrm = architecture(checkpoint["config"]) == "vitrm"
        self.precision = checkpoint["config"]["training"]["precision"] if self.device.type == "cuda" else "float32"
        self.refinement_records = []
        if self.device.type != "cuda" and not self.is_vitrm:
            raise RuntimeError("Experiment 2 TRM inference requires CUDA FP16 AMP")
        self.inference_config = {
            "uses_prompt": False, "training_regime": "supervised", "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": file_sha256(checkpoint_path), "upstream_sha": checkpoint["upstream_sha"],
            "architecture": architecture(checkpoint["config"]),
            "implementation": checkpoint.get("implementation"), "paper_url": checkpoint.get("paper_url"),
            "device": str(self.device), "dtype": "float16_autocast" if self.precision == "float16" else "float32", "parameter_dtype": "float32", "preprocessing": self.preprocessing,
            "model": checkpoint["config"]["model"], "split_digest": checkpoint["split_digest"],
            "parameter_counts": model.parameter_counts(), "refinement_steps": model.config["refinement_steps"],
            "selected_epoch": checkpoint["epoch"], "training_status": checkpoint["training_status"],
            "training_session": checkpoint["training_session"],
        }

    def generate(self, image_path, prompt):
        patches = load_image_patches(image_path, self.preprocessing).unsqueeze(0).to(self.device)
        if self.is_vitrm and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        with torch.inference_mode(), autocast_context(self.device, self.precision):
            steps = self.model.forward_steps(patches) if self.is_vitrm else None
            logits = steps[:, -1] if self.is_vitrm else self.model(patches)
        if self.is_vitrm and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = perf_counter() - started
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite TRM logits")
        if self.is_vitrm:
            if not torch.isfinite(steps).all():
                raise FloatingPointError("Non-finite ViTRM refinement logits")
            probabilities = steps[0].float().softmax(-1).cpu()
            predictions = steps[0].argmax(-1).cpu().tolist()
            for index, values in enumerate(probabilities):
                self.refinement_records.append({"image_path": str(image_path), "step": index + 1,
                    "predicted_severity": predictions[index], "model_seconds_all_steps": elapsed,
                    **{f"probability_{i}": float(value) for i, value in enumerate(values)}})
        return json.dumps(severity_prediction(int(logits.argmax(-1).item())))


def export_refinements(adapter, test_frame, run):
    """Per-step metrics with explicit planned coverage, including failed images."""
    from .evaluation import evaluate
    from .image_quality_pipeline import InferenceResult
    labels = {row.file_path: {key: getattr(row, key) for key in ("noise", "severity", "route")}
              for row in test_frame.itertuples()}
    records = pd.DataFrame(adapter.refinement_records, columns=["image_path", "step", "predicted_severity",
        "model_seconds_all_steps", *[f"probability_{i}" for i in range(4)]])
    records["true_severity"] = records.image_path.map(lambda path: int(labels[path]["severity"]))
    records.to_csv(run / "refinement_predictions.csv", index=False)
    summaries = []
    for step in range(1, adapter.model.config["refinement_steps"] + 1):
        group = records.loc[records.step == step]
        results = [InferenceResult(image_path=row.image_path, raw_output="",
            prediction=severity_prediction(int(row.predicted_severity)), error=None) for row in group.itertuples()]
        metrics = evaluate(results, labels)
        summaries.append({"step": step, "planned": len(labels), "valid": len(group),
            "coverage": len(group) / len(labels),
            "severity_accuracy_valid": metrics["severity"]["accuracy"],
            "severity_correct_planned": metrics["severity"]["correct"] / len(labels),
            "severity_macro_f1": metrics["severity"]["macro_f1"],
            "severity_mae": metrics["severity"]["mae"],
            "route_accuracy_valid": metrics["route"]["accuracy"],
            "route_false_negatives": metrics["route"]["confusion_matrix"][1][0],
            "route_false_positives": metrics["route"]["confusion_matrix"][0][1]})
    pd.DataFrame(summaries).to_csv(run / "refinement_summary.csv", index=False)


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
                  response_format="json", model_name="VisionViTRM" if adapter.is_vitrm else "VisionTRM", show_progress=show_progress)
    run = session / "experiment_002/run_001"
    if adapter.is_vitrm:
        export_refinements(adapter, test_frame, run)
    return run
