"""Offline task metrics and repeated-inference decision stability.

Confusion matrices use true labels as rows and predicted labels as columns.
Binary precision/recall use positive class 1; severity uses macro averaging.
Undefined per-class ratios are zero, including classes absent from a run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .image_quality_pipeline import (
    IMAGE_QUALITY_PROMPT, TASKS, VALID_VALUES, InferenceResult,
    _normalize_labels, _positive_integer, _write_json, load_results,
    run_inference, validate_prediction,
)


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


def decision_stability(passes: Sequence[Sequence[InferenceResult]], *, image_paths=None) -> dict[str, Any]:
    """Compare independent passes, aligning by exact image path.

    Stable means every requested pass returned a valid, identical decision.
    Pairwise agreement uses valid pairs only, and is null if none exist.
    Missing/error responses never count as agreement. No accuracy is implied.
    """
    if len(passes) < 2:
        raise ValueError('Stability requires at least two independent passes')
    indexed = []
    for results in passes:
        index = {}
        for result in results:
            if result.image_path in index:
                raise ValueError(f'Duplicate image path in a pass: {result.image_path}')
            if result.prediction is not None:
                result = InferenceResult(result.image_path, result.raw_output,
                                         validate_prediction(result.prediction), result.error)
            index[result.image_path] = result
        indexed.append(index)
    paths = sorted(set(image_paths) if image_paths is not None else set().union(*(set(p) for p in indexed)))
    if any(set(p) - set(paths) for p in indexed):
        raise ValueError('Results contain images outside the expected evaluation set')
    report = {'passes': len(passes), 'total_images': len(paths), 'tasks': {}}
    for task in (*TASKS, 'joint'):
        details = []
        agreed = pairs = complete = stable = changed = 0
        for path in paths:
            values = []
            for index in indexed:
                result = index.get(path)
                valid = result is not None and result.prediction is not None and result.error is None
                values.append((tuple(result.prediction[t] for t in TASKS) if task == 'joint'
                               else result.prediction[task]) if valid else None)
            counts = Counter(value for value in values if value is not None)
            valid_count = sum(counts.values())
            pair_count = valid_count * (valid_count - 1) // 2
            agreement_count = sum(n * (n - 1) // 2 for n in counts.values())
            is_complete = valid_count == len(passes)
            is_stable = is_complete and len(counts) == 1
            complete += is_complete
            stable += is_stable
            changed += len(counts) > 1
            pairs += pair_count
            agreed += agreement_count
            details.append({'image_path': path, 'decisions': [list(v) if isinstance(v, tuple) else v for v in values],
                            'valid_passes': valid_count,
                            'complete': is_complete, 'stable': is_stable,
                            'pairwise_agreement': agreement_count / pair_count if pair_count else None})
        report['tasks'][task] = {
            'complete_images': complete, 'incomplete_images': len(paths) - complete,
            'stable_images': stable, 'changed_images': changed,
            'coverage': complete / len(paths) if paths else 0.0,
            'stable_fraction': stable / len(paths) if paths else None,
            'stable_fraction_complete': stable / complete if complete else None,
            'valid_pairs': pairs, 'agreeing_pairs': agreed,
            'pairwise_agreement': agreed / pairs if pairs else None,
            'per_image': details,
        }
    return report


def evaluate_stability(run_dirs: Sequence[str | Path]) -> dict[str, Any]:
    """Compare saved passes with identical model, prompt, holdout and labels."""
    runs = [Path(path) for path in run_dirs]
    if len(runs) < 2:
        raise ValueError('Stability requires at least two independent passes')
    if len({p.resolve() for p in runs}) != len(runs):
        raise ValueError('Stability requires distinct run directories')
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in runs]
    labels = [json.loads((p / 'labels.json').read_text()) for p in runs]
    keys = ('experiment_number', 'model_name', 'adapter', 'prompt', 'inference_mode',
            'examples', 'held_out_subjects', 'batch_size', 'total_images', 'inference_config')
    for manifest, truth in zip(manifests[1:], labels[1:]):
        if truth != labels[0] or any(manifest.get(k) != manifests[0].get(k) for k in keys):
            raise ValueError('Stability passes must use identical labels and inference configuration')
    report = decision_stability([load_results(p) for p in runs], image_paths=labels[0])
    report['run_dirs'] = [str(p.resolve()) for p in runs]
    report['run_statuses'] = [m['status'] for m in manifests]
    return report


@dataclass(frozen=True)
class EvaluationConfig:
    """Saved-run location and repeat count, matching inference configuration."""
    output_dir: str | Path = 'outputs'
    experiment_number: int = 1
    stability_passes: int = 3

    def __post_init__(self):
        _positive_integer(self.experiment_number, 'experiment_number')
        _positive_integer(self.stability_passes, 'stability_passes')
        if self.stability_passes < 2:
            raise ValueError('stability_passes must be at least 2')


def run_stability_experiment(model, image_paths, *, labels, config=None,
                             prompt=IMAGE_QUALITY_PROMPT, **inference_options):
    """Run fresh inference for each pass; checkpoint runs and a group report.

    Options such as batch_size, inference_mode, examples and model_name are
    forwarded unchanged. The model is reused, but predictions are never reused.
    Returns the report path, metrics for each pass, and stability measurements.
    """
    config = config or EvaluationConfig()
    forbidden = {'output_dir', 'experiment_number', 'run_number'} & inference_options.keys()
    if forbidden:
        raise ValueError(f'Configure run locations using EvaluationConfig, not {sorted(forbidden)}')
    paths = [str(path) for path in image_paths]
    if not paths:
        raise ValueError('Stability requires at least one evaluation image')
    if len(set(paths)) != len(paths):
        raise ValueError('Each pass must contain unique image paths')
    truth = _normalize_labels(labels)
    if set(paths) - truth.keys():
        raise ValueError('Every input image requires ground truth for saved evaluation')
    # Keep labels scoped to requested inputs; run_inference additionally removes holdouts.
    if hasattr(labels, 'columns'):
        labels = labels.loc[labels['file_path'].astype(str).isin(paths)].copy()
    else:
        labels = {path: truth[path] for path in paths}
    # Validate the effective evaluation set before allocating any output folders.
    examples = tuple(inference_options.get('examples', ()))
    held_out = {str(example.subject_id) for example in examples}
    example_paths = {str(Path(example.image_path).resolve()) for example in examples}
    subject_by_path = (dict(zip(labels['file_path'].astype(str), labels['subject_id'].astype(str)))
                       if hasattr(labels, 'columns') and 'subject_id' in labels else {})
    if not any(subject_by_path.get(path, Path(path).stem) not in held_out
               and str(Path(path).resolve()) not in example_paths for path in paths):
        raise ValueError('Holding out examples leaves no evaluation images')
    if 'examples' in inference_options:
        inference_options['examples'] = examples
    experiment = Path(config.output_dir) / f'experiment_{config.experiment_number:03d}'
    experiment.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        group = experiment / f'stability_{number:03d}'
        try:
            group.mkdir()
            break
        except FileExistsError:
            number += 1
    state = {'schema_version': 1, 'status': 'running', 'requested_passes': config.stability_passes,
             'run_dirs': [], 'metrics': []}
    try:
        for _ in range(config.stability_passes):
            run_number = 1
            while (experiment / f'run_{run_number:03d}').exists():
                run_number += 1
            run = experiment / f'run_{run_number:03d}'
            state['run_dirs'].append(str(run.resolve()))
            _write_json(group / 'report.json', state)
            run_inference(model, paths, prompt, labels=labels,
                          output_dir=config.output_dir, experiment_number=config.experiment_number,
                          run_number=run_number, **inference_options)
            state['metrics'].append(evaluate_saved_run(run, save=False))
            _write_json(group / 'report.json', state)
        state['stability'] = evaluate_stability(state['run_dirs'])
        state['status'] = 'completed'
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                     error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json(group / 'report.json', state)
    return {'report_path': str((group / 'report.json').resolve()), **state}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', type=Path)
    parser.add_argument('--stability', action='store_true', help='Compare runs as repeated passes')
    parser.add_argument('--report', type=Path, help='Optional path for the combined JSON report')
    args = parser.parse_args()
    report = (evaluate_stability(args.run_dirs) if args.stability else
              {str(run): evaluate_saved_run(run) for run in args.run_dirs})
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
