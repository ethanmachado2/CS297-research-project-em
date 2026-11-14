"""Shared dataset labels, strict responses, and incremental inference records."""

from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import islice
from time import perf_counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import pandas as pd
from tqdm.auto import tqdm

FEW_SHOT_GUIDANCE = """

The four labeled reference images define this dataset's clean (0), weak (1),
medium (2), and strong (3) augmentation levels. Use their visible grain strength,
speckle contrast, and texture disruption to calibrate the descriptions above. A strong
reference may still show a clear face. Compare noise texture, not identity,
lighting, skin tone, or background content. If each reference has an overview
and a detail view, compare target details with reference details at that scale.
Choose the closest reference LEVEL for the final target image, allowing for
different image content. The references are calibration examples, not clean
versions of the target. Return only the target's JSON using the exact mapping
above; do not return reference predictions or describe comparisons.
"""

IMAGE_QUALITY_PROMPT = (Path(__file__).resolve().parents[1] / "configs/qwen_prompt.txt").read_text()

INFERENCE_MODES = ("zero_shot", "few_shot", "supervised")

TASKS = ("noise", "severity", "route")

EXPECTED_KEYS = frozenset(TASKS)

VALID_VALUES = {
    "noise": frozenset({"0", "1"}),
    "severity": frozenset({"0", "1", "2", "3"}),
    "route": frozenset({"0", "1"}),
}

@dataclass(frozen=True)
class InferenceResult:
    image_path: str
    raw_output: str
    prediction: dict[str, str] | None
    error: str | None = None

@dataclass(frozen=True)
class FewShotExample:
    image_path: str
    subject_id: str
    prediction: dict[str, str]

def prepare_few_shot_split(
    dataset: pd.DataFrame, example_subject: str = "0",
) -> tuple[pd.DataFrame, list[FewShotExample]]:
    """Reserve one subject's four severity examples and exclude all its rows.

    By default, filename stems identify subjects across augmentation folders.
    Supply a ``subject_id`` column to group multiple filenames per subject.
    Use the returned evaluation frame for both modes for a fair comparison.
    """
    subject_ids = (dataset["subject_id"].astype(str) if "subject_id" in dataset
                   else dataset["file_path"].map(lambda path: Path(path).stem))
    selected = dataset.loc[subject_ids == str(example_subject)]
    examples = []
    for severity in ("0", "1", "2", "3"):
        candidates = selected.loc[selected["severity"].astype(str) == severity].sort_values("file_path")
        if candidates.empty:
            raise ValueError(f"Example subject {example_subject!r} is missing severity {severity}")
        row = candidates.iloc[0]
        examples.append(FewShotExample(str(row.file_path), str(example_subject),
                                       validate_prediction({task: row[task] for task in TASKS})))
    evaluation = dataset.loc[subject_ids != str(example_subject)].copy()
    if evaluation.empty:
        raise ValueError("Holding out the example subject leaves no evaluation images")
    return evaluation, examples

def build_dataset(data_root: str | Path) -> pd.DataFrame:
    """Build the common labeled dataset from ``data/files``.

    Labels are strings because the model contract is a JSON string-label
    contract.  ``augmentation/1``, ``/2``, and ``/3`` are weak, medium, and
    strong respectively; the unaugmented images are clean.
    """

    root = Path(data_root)
    groups = (
        (root / "images", "none", "0", "0", "0"),
        (root / "augmentation" / "1", "weak", "1", "1", "0"),
        (root / "augmentation" / "2", "medium", "2", "1", "1"),
        (root / "augmentation" / "3", "strong", "3", "1", "1"),
    )
    rows: list[dict[str, str]] = []
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    for folder, augmentation, severity, noise, route in groups:
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                rows.append(
                    {
                        "file_path": str(path),
                        "file_name": path.name,
                        "subject_id": path.stem,
                        "augmentation": augmentation,
                        "noise": noise,
                        "severity": severity,
                        "route": route,
                    }
                )
    return pd.DataFrame(rows, columns=["file_path", "file_name", "subject_id", "augmentation", "noise", "severity", "route"])

def validate_prediction(value: Mapping[str, Any]) -> dict[str, str]:
    """Validate and normalize one model response against the JSON contract."""

    if set(value) != EXPECTED_KEYS:
        missing = EXPECTED_KEYS.difference(value)
        extra = set(value).difference(EXPECTED_KEYS)
        raise ValueError(f"prediction keys must be exactly {sorted(EXPECTED_KEYS)}; missing={sorted(missing)}, extra={sorted(extra)}")
    prediction = {key: str(value[key]) for key in TASKS}
    for key, allowed in VALID_VALUES.items():
        if prediction[key] not in allowed:
            raise ValueError(f"{key} must be one of {sorted(allowed)}, got {prediction[key]!r}")
    expected_route = "1" if prediction["severity"] in {"2", "3"} else "0"
    if prediction["route"] != expected_route:
        raise ValueError(f"route {prediction['route']!r} does not follow severity {prediction['severity']!r}")
    if prediction["severity"] == "0" and prediction["noise"] != "0":
        raise ValueError("severity 0 requires noise 0")
    if prediction["severity"] in {"1", "2", "3"} and prediction["noise"] != "1":
        raise ValueError("non-zero severity requires noise 1")
    return prediction

def parse_prediction(raw_output: str) -> dict[str, str]:
    """Parse exactly one JSON object, rejecting prose and extra JSON keys."""

    text = raw_output.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model output is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("model output must be one JSON object")
    return validate_prediction(value)

def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")

def _write_json(path: Path, value: Any) -> None:
    """Replace a JSON file only after its complete contents are written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)

def _create_run(output_dir: str | Path, experiment_number: int, run_number: int | None) -> Path:
    experiment = Path(output_dir) / f"experiment_{experiment_number:03d}"
    experiment.mkdir(parents=True, exist_ok=True)
    if run_number is not None:
        run = experiment / f"run_{run_number:03d}"
        run.mkdir()  # Never replace an existing experiment run.
        return run
    number = 1
    while True:
        run = experiment / f"run_{number:03d}"
        try:
            run.mkdir()
            return run
        except FileExistsError:
            number += 1

def run_inference(
    model: Any,
    image_paths: Iterable[str | Path],
    prompt: str = IMAGE_QUALITY_PROMPT,
    *,
    batch_size: int = 1,
    output_dir: str | Path | None = "outputs",
    experiment_number: int = 1,
    run_number: int | None = None,
    show_progress: bool = True,
    labels: Mapping[str, Mapping[str, str]] | pd.DataFrame | None = None,
    model_name: str | None = None,
    inference_mode: str = "zero_shot",
    examples: Sequence[FewShotExample] = (),
    response_format: str = "json",
) -> list[InferenceResult]:
    """Infer, checkpoint each batch, and optionally persist ground truth/metrics.

    Adapters with ``generate_batch(paths, prompt)`` perform one model call per
    batch and must return one string per input, in order. Legacy adapters still
    work sequentially. Batch errors are recorded for every affected image.
    ``inference_mode`` selects the original zero-shot prompt or multimodal
    few-shot reference turns. Supplied examples are held out in either mode;
    pass the same examples to both modes to compare the same evaluation set.
    Set ``output_dir=None`` to disable persistence; existing run folders raise
    ``FileExistsError``. Automatically allocated run numbers start at one.
    ``response_format="severity_digit"`` accepts only a single 0–3 digit and
    derives noise/route from it, preserving the original raw response.
    """
    _positive_integer(batch_size, "batch_size")
    if response_format not in {"json", "severity_digit"}:
        raise ValueError("response_format must be json or severity_digit")
    if response_format == "severity_digit" and inference_mode != "zero_shot":
        raise ValueError("severity_digit currently supports zero_shot only")
    _positive_integer(experiment_number, "experiment_number")
    if run_number is not None:
        _positive_integer(run_number, "run_number")
    if inference_mode not in INFERENCE_MODES:
        raise ValueError(f"inference_mode must be one of {INFERENCE_MODES}")
    examples = tuple(examples)
    if inference_mode == "supervised" and examples:
        raise ValueError("supervised mode does not accept examples")
    if inference_mode == "few_shot" and not examples:
        raise ValueError("few_shot mode requires one subject's four labeled image examples")
    if examples:
        if len({str(example.subject_id) for example in examples}) != 1:
            raise ValueError("Few-shot examples must belong to exactly one subject")
        examples = tuple(FewShotExample(str(example.image_path), str(example.subject_id),
                                        validate_prediction(example.prediction)) for example in examples)
        predictions = [example.prediction for example in examples]
        if len(examples) != 4 or {p["severity"] for p in predictions} != {"0", "1", "2", "3"}:
            raise ValueError("Provide exactly one example for each severity: 0, 1, 2, 3")
    if inference_mode == "few_shot":
        if not getattr(model, "supports_few_shot", False):
            raise TypeError("Few-shot mode requires an adapter with supports_few_shot=True and examples support")
        prompt += FEW_SHOT_GUIDANCE
    paths = [str(path) for path in image_paths]
    truth = _normalize_labels(labels) if labels is not None else None
    subject_by_path = {}
    if isinstance(labels, pd.DataFrame) and "subject_id" in labels:
        subject_by_path = dict(zip(labels["file_path"].astype(str), labels["subject_id"].astype(str)))
    held_out_subjects = {str(example.subject_id) for example in examples}
    example_paths = {str(Path(example.image_path).resolve()) for example in examples}

    def is_held_out(path: str) -> bool:
        return (subject_by_path.get(path, Path(path).stem) in held_out_subjects
                or str(Path(path).resolve()) in example_paths)

    excluded_paths = [path for path in paths if is_held_out(path)]
    paths = [path for path in paths if not is_held_out(path)]
    if truth is not None:
        truth = {path: values for path, values in truth.items() if not is_held_out(path)}
    run = _create_run(output_dir, experiment_number, run_number) if output_dir is not None else None
    manifest = {
        "schema_version": 1,
        "experiment_number": experiment_number,
        "run_number": int(run.name.removeprefix("run_")) if run else run_number,
        "model_name": model_name,
        "adapter": type(model).__name__,
        "inference_config": getattr(model, "inference_config", None),
        "prompt": prompt,
        "response_format": response_format,
        "inference_mode": inference_mode,
        "examples": [asdict(example) for example in examples],
        "held_out_subjects": sorted(held_out_subjects),
        "excluded_image_paths": excluded_paths,
        "batch_size": batch_size,
        "batched_generation": callable(getattr(model, "generate_batch", None)),
        "total_images": len(paths),
        "completed_images": 0,
        "completed_batches": 0,
        "failed_images": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    results: list[InferenceResult] = []
    started = perf_counter()
    iterator = iter(paths)
    try:
        if run:
            _write_json(run / "manifest.json", manifest)
            if truth is not None:
                _write_json(run / "labels.json", truth)
        with tqdm(total=len(paths), desc="Inference", unit="image", disable=not show_progress) as progress:
            for batch_number, batch in enumerate(iter(lambda: list(islice(iterator, batch_size)), []), 1):
                batch_started = perf_counter()
                batch_results = []
                if callable(getattr(model, "generate_batch", None)):
                    try:
                        outputs = (model.generate_batch(batch, prompt, examples=examples)
                                   if inference_mode == "few_shot" else model.generate_batch(batch, prompt))
                        if isinstance(outputs, (str, bytes)):
                            raise ValueError("generate_batch must return one response per image, not a string")
                        outputs = list(outputs)
                        if len(outputs) != len(batch):
                            raise ValueError(f"generate_batch returned {len(outputs)} responses for {len(batch)} images")
                    except Exception as exc:
                        batch_results = [InferenceResult(path, "", None, str(exc)) for path in batch]
                    else:
                        batch_results = [_parse_result(path, raw, response_format) for path, raw in zip(batch, outputs)]
                else:
                    for path in batch:
                        try:
                            if inference_mode == "few_shot":
                                raw = model.generate(path, prompt, examples=examples)
                            else:
                                raw = model.generate(path, prompt) if hasattr(model, "generate") else model(path, prompt)
                        except Exception as exc:
                            batch_results.append(InferenceResult(path, "", None, str(exc)))
                        else:
                            batch_results.append(_parse_result(path, raw, response_format))
                if run:
                    _write_json(run / f"batch_{batch_number:05d}.json", {
                        "schema_version": 1,
                        "experiment_number": experiment_number,
                        "run_number": manifest["run_number"],
                        "batch_number": batch_number,
                        "inference_mode": inference_mode,
                        "elapsed_seconds": perf_counter() - batch_started,
                        "results": [asdict(result) for result in batch_results],
                    })
                results.extend(batch_results)
                manifest.update(completed_images=len(results), completed_batches=batch_number,
                                failed_images=manifest["failed_images"] + sum(result.error is not None for result in batch_results))
                if run:
                    _write_json(run / "manifest.json", manifest)
                progress.update(len(batch))
                progress.set_postfix(batch=batch_number, failed=manifest["failed_images"])
        if run and truth is not None:
            _write_json(run / "metrics.json", evaluate(results, truth))
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest.update(finished_at=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=perf_counter() - started)
        if run:
            _write_json(run / "manifest.json", manifest)
    return results

def _parse_result(path: str, raw: Any, response_format: str = "json") -> InferenceResult:
    raw = str(raw)
    try:
        if response_format == "severity_digit":
            severity = raw.strip()
            if severity not in {"0", "1", "2", "3"}:
                raise ValueError("Expected exactly one severity digit: 0, 1, 2, or 3")
            return InferenceResult(path, raw, validate_prediction({
                "severity": severity,
                "noise": "0" if severity == "0" else "1",
                "route": "1" if severity in {"2", "3"} else "0",
            }))
        return InferenceResult(path, raw, parse_prediction(raw))
    except Exception as exc:
        return InferenceResult(path, raw, None, str(exc))

def load_results(run_dir: str | Path) -> list[InferenceResult]:
    """Reload completed batch files, including those from interrupted runs."""
    run = Path(run_dir)
    if not run.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run}")
    results = []
    for path in sorted(run.glob("batch_*.json"), key=lambda path: int(path.stem.removeprefix("batch_"))):
        payload = json.loads(path.read_text(encoding="utf-8"))
        results.extend(InferenceResult(**record) for record in payload["results"])
    return results

def _normalize_labels(labels: Mapping[str, Mapping[str, str]] | pd.DataFrame) -> dict[str, dict[str, str]]:
    if isinstance(labels, pd.DataFrame):
        if labels["file_path"].astype(str).duplicated().any():
            raise ValueError("Ground truth must contain unique image paths")
        truth = {str(row.file_path): {task: str(getattr(row, task)) for task in TASKS} for row in labels.itertuples()}
    else:
        truth = {str(path): {task: str(values[task]) for task in TASKS} for path, values in labels.items()}
        if len(truth) != len(labels):
            raise ValueError("Ground truth must contain unique string image paths")
    return {path: validate_prediction(values) for path, values in truth.items()}

def evaluate(results: Sequence[InferenceResult], labels: Mapping[str, Mapping[str, str]] | pd.DataFrame) -> dict[str, Any]:
    """Compute the shared task metrics; see :mod:`src.evaluation` for details."""
    from .evaluation import evaluate as evaluate_metrics
    return evaluate_metrics(results, labels)
