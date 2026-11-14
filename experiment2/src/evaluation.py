"""Task metrics with explicit valid-prediction coverage."""

from __future__ import annotations

import json

from pathlib import Path

from typing import Any

from .image_quality_pipeline import TASKS, VALID_VALUES, _normalize_labels, _write_json, load_results, validate_prediction

def evaluate(results, labels) -> dict[str, Any]:
    """Score valid, labeled responses; retain failure and label coverage counts.

    Accuracy/MAE denominators contain only valid labeled predictions. Coverage
    is relative to supplied results; label_coverage exposes missing saved batches.
    Repeated results must be evaluated as separate passes, not pooled here.
    """
    truth = _normalize_labels(labels)
    seen = set()
    considered = []
    failed = unlabeled = 0
    for result in results:
        if result.image_path in seen:
            raise ValueError(f"Duplicate image path in a single pass: {result.image_path}")
        seen.add(result.image_path)
        if result.prediction is None or result.error is not None:
            failed += 1
            continue
        prediction = validate_prediction(result.prediction)
        if result.image_path not in truth:
            unlabeled += 1
            continue
        considered.append((result.image_path, prediction))
    metrics = {}
    for task in TASKS:
        classes = sorted(VALID_VALUES[task])
        matrix = [[0 for _ in classes] for _ in classes]
        for path, prediction in considered:
            matrix[classes.index(truth[path][task])][classes.index(prediction[task])] += 1
        per_class = {}
        for i, label in enumerate(classes):
            tp = matrix[i][i]
            support = sum(matrix[i])
            predicted = sum(row[i] for row in matrix)
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            per_class[label] = {
                "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "support": support,
            }
        count = len(considered)
        correct = sum(matrix[i][i] for i in range(len(classes)))
        macro_precision = sum(v['precision'] for v in per_class.values()) / len(classes)
        macro_recall = sum(v['recall'] for v in per_class.values()) / len(classes)
        binary = task != 'severity'
        metrics[task] = {
            "correct": correct, "evaluated": count, "total_results": len(results),
            "failed": failed, "unlabeled": unlabeled, "total_labels": len(truth),
            "coverage": count / len(results) if results else 0.0,
            "label_coverage": count / len(truth) if truth else 0.0,
            "accuracy": correct / count if count else 0.0,
            "precision": per_class['1']['precision'] if binary else macro_precision,
            "recall": per_class['1']['recall'] if binary else macro_recall,
            "precision_recall_average": "binary" if binary else "macro",
            "macro_precision": macro_precision, "macro_recall": macro_recall,
            "macro_f1": sum(v['f1'] for v in per_class.values()) / len(classes),
            "labels": classes, "confusion_matrix": matrix, "per_class": per_class,
        }
        if task == 'severity':
            metrics[task]['mae'] = (sum(abs(int(prediction[task]) - int(truth[path][task]))
                                         for path, prediction in considered) / count if count else None)
        if task == 'route':
            metrics[task]['routing_accuracy'] = metrics[task]['accuracy']
    return metrics

def evaluate_saved_run(run_dir: str | Path, *, save: bool = True) -> dict[str, Any]:
    """Recompute metrics from saved batches and labels without loading a model."""
    run = Path(run_dir)
    labels = json.loads((run / 'labels.json').read_text(encoding='utf-8'))
    metrics = evaluate(load_results(run), labels)
    if save:
        _write_json(run / 'metrics.json', metrics)
    return metrics
