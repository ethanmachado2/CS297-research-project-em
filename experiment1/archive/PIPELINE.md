# Image-quality classification pipeline

The reusable implementation lives in [`src/image_quality_pipeline.py`](src/image_quality_pipeline.py). It separates the model-specific inference adapter from the common prompt, output contract, dataset labels, and evaluation logic.

## Shared prompt and model adapter

Every vision model receives the same `IMAGE_QUALITY_PROMPT`. A model can be Qwen, another Transformers model, or an API-backed model. It only needs to implement either:

```python
class MyVisionModel:
    def generate(self, image_path, prompt):
        # Convert image_path/prompt to the provider's request format.
        # Return the model's text response.
        return raw_text
```

or a callable with the same signature: `runner(image_path, prompt) -> str`.

The adapter must return the complete response text. `run_inference` then parses and validates exactly one JSON object, including the route/severity relationship. Invalid responses are retained as failed `InferenceResult` records instead of silently becoming labels.

## Three experiment blocks

```python
from src.image_quality_pipeline import build_dataset, evaluate, run_inference, task_view

df = build_dataset("data/files")

# The same inference batch supports all three evaluations.
results = run_inference(model_adapter, df["file_path"].tolist(), labels=df)
metrics = evaluate(results, df)

# Task-specific tables for separate experiment reporting:
noise_eval = task_view(df, "noise")       # binary: 0 clean, 1 augmented
severity_eval = task_view(df, "severity") # multiclass: 0/1/2/3
route_eval = task_view(df, "route")       # binary routing decision: 0/1
```

The resulting metrics include accuracy, precision, recall, Macro-F1, per-class scores, and a confusion matrix for `noise`, `severity`, and `route`. Severity additionally reports `mae`; routing includes `routing_accuracy`. Counts and coverage expose failed, missing, or unlabeled predictions. The same results can be compared across model adapters without changing the evaluation code.

## Adapter examples

For a provider whose SDK returns text:

```python
def api_adapter(image_path, prompt):
    response = client.responses.create(model="your-vision-model", input=...)
    return response.output_text
```

For a Transformers model, put its processor/model-specific image preparation in `generate`; the pipeline does not depend on Qwen classes or Qwen naming. Keep generation deterministic where supported (`temperature=0`) and use the prompt unchanged for cross-model comparisons.

## Batched inference, saved runs, and progress

The local notebook now imports `TransformersImageModel` from
[`src/transformers_adapter.py`](src/transformers_adapter.py); generation stays in
supporting Python scripts. Wrap your already loaded processor and model:

```python
from src.transformers_adapter import TransformersImageModel
from src.image_quality_pipeline import load_results

adapter = TransformersImageModel(model, processor, max_new_tokens=128)
results = run_inference(
    adapter,
    df["file_path"],
    batch_size=4,
    experiment_number=1,
    run_number=None,
    output_dir="outputs",
    labels=df,
    model_name="Qwen/Qwen3.5-0.8B",
    show_progress=True,
)
```

`experiment_number` identifies the experimental configuration; all three task
evaluations share it. `run_number=None` allocates the first unused positive run
number, starting at 1. An explicit existing run number raises `FileExistsError`
before generation begins. Experiment, run, and batch numbers are one-based:

```text
outputs/
  experiment_001/
    run_001/
      manifest.json
      labels.json
      batch_00001.json
      batch_00002.json
      metrics.json
```

Each batch is saved immediately, including experiment/run/batch identifiers,
elapsed time, image paths, raw responses, validated predictions, and errors.
Writes use a temporary file followed by replacement. The manifest records the
prompt, supplied model name, adapter inference settings (when exposed), batch
configuration, timestamps, counts, elapsed
time, and run status (`running`, `completed`, `interrupted`, or `failed`).
`labels.json` and final `metrics.json` are written when `labels` is supplied;
metrics include noise, severity, and routing. A completed run can include model
or parsing errors; inspect `failed_images` and evaluation coverage.

Completed batches survive interruption and can be evaluated later without model
inference. Reload an exact run directory (this does not resume generation):

```python
import json
from pathlib import Path

run = Path("outputs/experiment_001/run_001")
saved_results = load_results(run)
saved_labels = json.loads((run / "labels.json").read_text())
metrics = evaluate(saved_results, saved_labels)
```

The progress bar uses `tqdm.auto` for notebooks and terminals, reporting image
counts, estimated time remaining, the current batch, and failures. Disable it
with `show_progress=False`. `output_dir=None` disables saving. Existing calls
remain valid and use `batch_size=1`; they now save to `outputs` by default.

Adapters can optionally implement `generate_batch(image_paths, prompt) ->
Sequence[str]` to process multiple images in one model call. Responses must match
the input order and count. Legacy callables and `generate` adapters still work,
with sequential generation inside each saved batch. A failed batch call or wrong
response count records an error for every affected image; invalid JSON affects
only its own image. Reduce the batch size if device memory is exhausted.

Batching can accelerate **inference**, subject to hardware and model memory limits.
There is no training loop in this project. The Transformers adapter uses
independent image conversations, left padding, deterministic generation, evaluation
mode, and `torch.inference_mode()`. Its batched processing follows the
[Transformers multimodal chat interface](https://huggingface.co/docs/transformers/chat_templating_multimodal).
It targets decoder-only vision models; other providers can implement their own
batch adapters.

Run the focused checks without downloading model weights:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Zero-shot and few-shot inference modes

Set `INFERENCE_MODE = "zero_shot"` or `"few_shot"` in the local notebook's configuration cell; rerun from the split cell after switching modes.
`EXAMPLE_SUBJECT = "0"` selects the reference subject. Supporting Python scripts
handle the split, prompt selection, reference conversations, and saved metadata.

`build_dataset` assigns `subject_id` from the filename stem: `images/0.jpg` and
`augmentation/{1,2,3}/0.jpg` are treated as four versions of subject `0`. If your
dataset has multiple filenames for a person, supply an explicit `subject_id`
column with the correct grouping before preparing the split.

```python
from src.image_quality_pipeline import prepare_few_shot_split

evaluation_df, examples = prepare_few_shot_split(df, example_subject="0")
results = run_inference(
    adapter,
    evaluation_df["file_path"],
    labels=evaluation_df,
    inference_mode="few_shot",  # or "zero_shot"
    examples=examples,           # reserve the same subject in both modes
    batch_size=4,
    experiment_number=1,
    model_name="Qwen/Qwen3.5-0.8B",
)
```

For the current 200-image dataset, subject `0` supplies these four visual examples:

| Reference image | noise | severity | route |
| --- | --- | --- | --- |
| `data/files/images/0.jpg` | 0 | 0 | 0 |
| `data/files/augmentation/1/0.jpg` | 1 | 1 | 0 |
| `data/files/augmentation/2/0.jpg` | 1 | 2 | 1 |
| `data/files/augmentation/3/0.jpg` | 1 | 3 | 1 |

Few-shot mode extends the existing `IMAGE_QUALITY_PROMPT` with
`FEW_SHOT_GUIDANCE`. Each target conversation contains four reference image/user
turns, each followed by an assistant answer with the known JSON labels, then the
target image requesting its classification. With detail crops enabled, each image/user turn contains an overview plus a native center detail view. These are actual visual examples,
not filenames or text-only label demonstrations. Zero-shot mode uses the original
prompt and supplies only the target image.

`prepare_few_shot_split` excludes **all images of the reference subject** from
evaluation, leaving 196 images in the current dataset. It requires a reference
for every severity and rejects a split with no evaluation images. With multiple
images per subject/severity, it selects the first sorted image as the reference
but still excludes the entire subject. Pass the same `examples` in both modes:
zero-shot uses them only to reserve the identical holdout. Without examples,
zero-shot retains its previous behavior and evaluates all supplied image paths.

`run_inference` also filters reference subjects and example paths if the full
dataset is passed accidentally, and removes their ground truth from saved labels.
The run manifest records `inference_mode`, the reserved example paths and labels,
`held_out_subjects`, the effective prompt, and any excluded input image paths.
Each batch also records its mode. Separate automatically numbered runs can be
used to compare zero-shot and few-shot under one experiment number.

Custom few-shot adapters must declare `supports_few_shot = True` and accept the
keyword `examples` on `generate_batch` (or `generate`). Each example is a
`FewShotExample(image_path, subject_id, prediction)`. Unsupported adapters fail
before creating a run instead of silently dropping the reference images.
Few-shot prompts contain five images per target here, so a smaller batch size
may be appropriate than for zero-shot mode.


## Saved-run evaluation module

[`src/evaluation.py`](src/evaluation.py) computes the full metrics from saved
batches and `labels.json`, without loading model weights or images:

```python
from src.evaluation import evaluate_saved_run

metrics = evaluate_saved_run("outputs/experiment_001/run_001")
# Updates that run's metrics.json; use save=False for a read-only evaluation.
```

The existing `src.image_quality_pipeline.evaluate` entry point uses the same
implementation, so new inference runs automatically save the expanded metrics.
The local notebook displays each pass's metrics and confusion matrices and has
an independent saved-run evaluation cell.

- Noise and routing precision/recall use positive class `1` (noise present or
  denoising required). Severity precision/recall use macro averaging.
- Macro-F1 averages per-class F1 across the fixed labels: `0,1` for binary tasks
  and `0,1,2,3` for severity. Absent classes and zero denominators contribute zero.
  `macro_precision`, `macro_recall`, and `per_class` are also included.
- Confusion matrices have true labels on rows and predicted labels on columns,
  in the order in `labels`.
- Severity `mae` is mean absolute error on ordinal labels `0..3`, computed as
  `mean(abs(predicted - true))`. This is an evaluation loss, not a training step.
  It is `null` when there are no valid labeled predictions.
- Accuracy and MAE use valid labeled predictions only. `coverage` uses the
  number of supplied responses; `label_coverage` uses the number of saved labels,
  exposing incomplete batches in interrupted runs. `failed` and `unlabeled`
  count excluded responses. Duplicate image paths within a pass are rejected.
  Accuracy is zero when no valid labeled predictions exist; check `evaluated`.

Command-line evaluation (from the project root):

```sh
.venv/bin/python -m src.evaluation outputs/experiment_001/run_001
```

## Three-pass decision stability

Use the following instead of a single `run_inference` call for each experiment
configuration/prompting mode. Three independent passes are the default, as
configured in the local notebook. This makes three times as many image inference
requests as a single pass. Each call processes the images afresh with the same
model, prompt, batch size, and held-out subject.

```python
from src.evaluation import EvaluationConfig, run_stability_experiment
from src.image_quality_pipeline import prepare_few_shot_split

# Continue from df and adapter in the earlier examples.
evaluation_df, examples = prepare_few_shot_split(df, example_subject="0")
INFERENCE_MODE = "zero_shot"  # repeat separately with "few_shot"
report = run_stability_experiment(
    adapter,
    evaluation_df["file_path"],
    labels=evaluation_df,
    config=EvaluationConfig(
        output_dir="outputs", experiment_number=1, stability_passes=3,
    ),
    batch_size=4,
    inference_mode=INFERENCE_MODE,
    examples=examples,
    model_name="Qwen/Qwen3.5-0.8B",
)
print(report["report_path"])
```

Each pass allocates a normal `run_NNN` folder under `experiment_NNN`. An
independently numbered `stability_NNN/report.json` links the passes and stores
per-pass metrics, completion status, and stability results. Existing runs and
stability groups are preserved. Group state is checkpointed before and after
each pass so completed work remains discoverable after interruption.

Stability is measured separately for noise, severity, routing, and the joint
three-label decision. A stable image has valid, identical decisions in **all**
requested passes. Reports include stable/changed/incomplete image counts,
`stable_fraction` over all expected images, `stable_fraction_complete` over
fully valid images, and `pairwise_agreement` over all valid within-image pairs.
Missing or invalid predictions do not count as agreement. Pairwise agreement
is `null` when there are no valid pairs. Per-image decisions are retained for
inspection. Stability measures repeatability, independent of correctness.
Deterministic decoding remains enabled; this test measures repeatability under
the configured inference conditions and does not change sampling settings.

Existing independent passes can also be compared offline:

```python
from src.evaluation import evaluate_stability
stability = evaluate_stability([
    "outputs/experiment_001/run_001",
    "outputs/experiment_001/run_002",
    "outputs/experiment_001/run_003",
])
```

The comparison rejects duplicate run directories and mismatched saved labels,
experiment, model, adapter, prompt, mode, examples, holdout, batch size, or image
count, or exposed adapter inference settings. The Transformers adapter records
maximum new tokens, deterministic decoding, dtype, device, model source/revision,
and processor size. Custom adapters may expose a JSON-serializable
`inference_config` property; settings absent from saved metadata cannot be checked.
Older runs without that field can be compared to one another, but are rejected
when compared to runs that record different settings.

Interrupted passes remain eligible, with missing predictions reported as
incomplete. Keep zero-shot and few-shot in separate stability groups. Empty
stability experiments, including those with only held-out images, are rejected.
Duplicate ground-truth paths and invalid ground-truth labels are rejected before
inference rather than silently overwriting labels.

```sh
.venv/bin/python -m src.evaluation --stability \
  outputs/experiment_001/run_001 \
  outputs/experiment_001/run_002 \
  outputs/experiment_001/run_003 \
  --report outputs/experiment_001/stability_comparison.json
```


## End-to-end validation

The fast regression suite includes a complete integration check using generated
image files and a deterministic model fixture. It exercises both prompting modes,
three fresh passes, multiple batches, subject holdout, the Transformers adapter,
JSON parsing, saved results, task metrics, stability, and both CLI commands:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

For an opt-in **real Qwen model** check, use the locally cached snapshot:

```sh
.venv/bin/python scripts/validate_end_to_end.py \
  --subjects 2 --batch-size 1 --max-pixels 1048576 --detail-crop-size 512 \
  --output-dir outputs/end_to_end_new
```

Use a new output directory for each validation. The script uses only cached
weights and local images, selects MPS when available (otherwise CPU), and saves
`validation.json` plus ordinary runs and stability reports. It checks that every
response is valid, each pass has the expected images, saved metrics reload
identically, confusion-matrix counts reconcile, and CLI reports match Python
results. Omit `--subjects` to process all 196 evaluation images in each mode,
three times (1,176 target-image predictions). This may take substantially longer.

`--max-pixels` sets the overview processor budget; it does not modify source files.
The overview/detail configuration uses `MAX_PIXELS = 1048576` and
`DETAIL_CROP_SIZE = 512`. Each image is EXIF-oriented, converted to RGB, and
supplied as an overview plus a center crop taken directly from source pixels.
The 512×512 crop is patch-aligned and stays at native resolution under the
configured Qwen processor budget. References and targets use identical preparation.
The adapter rejects incompatible pixel budgets and images too small for the crop.
A center crop may miss localized degradation elsewhere; it does not preserve the
entire image at native resolution. Increasing only the overview budget still
averages away some grain, which is why the separate detail view is supplied.
Set `DETAIL_CROP_SIZE = None` for overview-only ablation runs. `MAX_PIXELS = None`
uses cached defaults (16,777,216 here), which can require much more memory.
Crop size, view mode, orientation handling, and processor budget are recorded in
`inference_config`; stability comparisons require these settings to match.

The shared zero-shot prompt now describes increasing grain/speckle contrast and
texture disruption instead of requiring unreadable facial features. Weak noise
can cover the image and be obvious at native scale. Few-shot guidance calibrates
these relative levels against the four labeled reference images. Exact numeric
augmentation thresholds cannot be recovered from the available labels alone.
All four allowed JSON combinations are shown explicitly, including severity 1
with route 0. Strict output validation and task definitions remain in place.

On October 20, 2026, the real-model workflow check completed on the Mac GPU with
four evaluation images (subject `1`, all four severities), reference subject `0`,
both modes, and three passes: **24 valid predictions, zero inference/parsing
failures**. Saved metrics and CLI/stability reports matched. Evidence is in
`outputs/end_to_end_smoke/validation.json`. That test used the 262,144-pixel cap
and preceded the subsequent duplicate-label, empty-evaluation, metadata, and
padding-call hardening. The later real-model rerun was declined at the GPU
permission prompt; the updated implementation is covered by the regression
suite. The recorded model predicted clean for all four test images in both
modes, giving routing accuracy 0.5 and severity MAE 1.5 despite perfect repeated
agreement. This validates the workflow on a small sample; it is not a full-dataset
performance result.


For the overview/detail configuration, the notebook previously retained `SMALL_TEST = False` and
`STABILITY_PASSES = 3`, and uses `BATCH_SIZE = 1` to limit memory with the extra
visual tokens. Set `SMALL_TEST = True` for a first check (four evaluation images),
and optionally `STABILITY_PASSES = 2` for a shorter repeatability check. Experiment 2
identifies the revised prompts and overview/detail input configuration. Small-test
outputs use `outputs/notebook_small_test/`; full runs use
`outputs/notebook_experiments/`. Old experiment outputs remain separate.
Restart the kernel and Run All after changing the shared Python modules. Changing
`MAX_PIXELS` requires processor/adapter recreation; changing `DETAIL_CROP_SIZE`
requires adapter recreation. The notebook's old cell outputs were cleared to
avoid displaying previous metrics beside the revised configuration.

## Full-resolution comparison

`TransformersImageModel(..., full_resolution=True)` disables image resizing,
EXIF-orients and converts each image to RGB, and edge-pads the right/bottom to
multiples of `patch_size * merge_size` (32 for this cached Qwen). The adapter
checks the processor's actual image grid against the padded source dimensions.
It preserves original pixels, performs no resampling, and cannot be combined
with detail crops. Normalization and patch extraction remain required. The
manifest records full-resolution mode, padding, and the effective resize switch.
The pixel budget is ignored in this mode. Full-resolution few-shot inference
includes all four reference images and may require substantially more memory.

The local notebook now starts with `FULL_RESOLUTION = True`, `SMALL_TEST = True`,
`BATCH_SIZE = 1`, and experiment 3. To isolate resolution from prompt changes,
run the separate single-pass comparison using the revised prompt in both arms:

```sh
.venv/bin/python scripts/test_full_resolution.py --subject 35
```

This uses cached weights on MPS, runs the four levels once with a 262,144-pixel
budget and once at full resolution, and checkpoints results/metrics under
`outputs/full_resolution_test/`. It preserves policy failures as recorded
failures; completion does not imply correct predictions. Use a new
`--output-dir` for another comparison. It explicitly requires GPU access by
default instead of silently starting a slow CPU run; `--device cpu` opts in to
CPU execution. See `comparison.json` and the per-batch files for results.

## Single-image prompt and resolution comparison

`notebooks/cs297-experiment1-em-local-single-image-comparison.ipynb` is a separate
pilot based on the earlier single-image, zero-shot configuration. It compares
the recovered original prompt, a shorter JSON prompt, severity-only digit
output, and the digit prompt with 4,194,304- and 2,097,152-pixel caps. All five
arms use the same balanced sample, model, greedy decoding and batch size 1,
with no detail crops. Defaults select five subjects (20 images), one pass per
arm. Outputs are checkpointed under a unique session in
`outputs/single_image_comparisons/`, with summary and failure CSVs.

`run_inference(..., response_format="severity_digit")` strictly accepts a single
digit 0–3 after whitespace removal and derives noise/route in Python. Original
raw responses are preserved; prose and invalid digits remain failures. The
default `response_format="json"` retains the existing behavior. The comparison
reports valid-only accuracy, coverage, and correct predictions per attempt so
format failures remain visible. It uses normal processor resizing, including
in the original-budget arms, rather than disabling resizing. Select a winning
arm with `ARM_NAMES` and set `SUBJECT_COUNT = None` for full-dataset evaluation.
