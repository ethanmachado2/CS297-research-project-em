"""Model-free saved-run reports, with the existing notebook's CSV conventions."""
from pathlib import Path
import json

import pandas as pd

from .image_quality_pipeline import TASKS, load_results
from .evaluation import evaluate_saved_run


def export_saved_run(run_dir):
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    labels = json.loads((run / "labels.json").read_text())
    results = load_results(run)
    metrics = evaluate_saved_run(run, save=True)
    rows = []
    for batch_path in sorted(run.glob("batch_*.json")):
        batch = json.loads(batch_path.read_text())
        for record in batch["results"]:
            prediction = record["prediction"] or {}
            truth = labels[record["image_path"]]
            row = {"image_path": record["image_path"], "file_name": Path(record["image_path"]).name,
                   "subject_id": Path(record["image_path"]).stem, "raw_output": record["raw_output"],
                   "error": record["error"], "valid": record["prediction"] is not None and record["error"] is None,
                   "elapsed_seconds": batch["elapsed_seconds"]}
            for task in TASKS:
                row.update({f"true_{task}": truth[task], f"predicted_{task}": prediction.get(task),
                            f"{task}_correct": row["valid"] and prediction.get(task) == truth[task]})
            rows.append(row)
    columns = ["image_path", "file_name", "subject_id", "raw_output", "error", "valid", "elapsed_seconds",
               *[name for task in TASKS for name in (f"true_{task}", f"predicted_{task}", f"{task}_correct")]]
    predictions = pd.DataFrame(rows, columns=columns)
    failures = predictions.loc[~predictions.valid.astype(bool)].copy()
    attempted, planned = len(results), manifest["total_images"]
    summary = []
    for task in TASKS:
        values = metrics[task]
        summary.append({"Task": task, "Correct": values["correct"], "Evaluated": values["evaluated"],
            "Accuracy (valid)": values["accuracy"], "Precision (valid)": values["precision"],
            "Recall (valid)": values["recall"], "Macro-F1 (valid)": values["macro_f1"],
            "MAE (valid)": values.get("mae"),
            "Correct / attempted": values["correct"] / attempted if attempted else None,
            "Correct / planned": values["correct"] / planned if planned else None})
    summary = pd.DataFrame(summary).set_index("Task")
    coverage = []
    for severity in ("0", "1", "2", "3"):
        planned_class = sum(value["severity"] == severity for value in labels.values())
        group = predictions.loc[predictions.true_severity == severity]
        valid = int(group.valid.sum())
        coverage.append({"True severity": severity, "Planned": planned_class, "Attempted": len(group),
            "Valid": valid, "Failed": len(group) - valid,
            "Coverage of planned": valid / planned_class if planned_class else None})
    coverage = pd.DataFrame(coverage).set_index("True severity")
    for name, frame in (("predictions", predictions), ("failures", failures),
                        ("evaluation_summary", summary), ("coverage_by_severity", coverage)):
        frame.to_csv(run / f"{name}.csv", index=name not in {"predictions", "failures"})
    return {"metrics": metrics, "predictions": predictions, "failures": failures, "summary": summary,
            "coverage": coverage, "status": manifest["status"], "planned": planned, "attempted": attempted,
            "complete": manifest["status"] == "completed" and attempted == planned,
            "inference_seconds": manifest.get("elapsed_seconds"),
            "training_status": (manifest.get("inference_config") or {}).get("training_status")}
