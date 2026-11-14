# CS297 Experiment 1: execution guide

This project classifies face-image degradation into three outputs: noise present,
severity, and whether to route the image to denoising. It predicts a routing
decision; it does not execute a denoiser.

For **Colab Enterprise or GCP Workbench**, start with the self-contained
[cloud notebooks and setup guide](notebooks/colab_enterprise/README.md). They live
separately from the local notebooks, prepare missing data, select CUDA, and upload
run artifacts to Cloud Storage. All workflows use the single `requirements.txt`;
the cloud installer retains the managed runtime's Torch and Jupyter packages.

This guide covers all three local experiment notebooks and the supporting TRM
diagnostic and validation runs outside `archive/`. Commands below run from the
**`experiment1` project root**, where this README, `src/`, and `notebooks/` live.
Replace example output names with a fresh name for each execution.

The Git repository contains source, configuration, documentation, tests, and
notebooks. Datasets, the model cache, generated results/checkpoints (including
`archive/outputs/`), and Google Drive document pointers remain local and are
excluded by `.gitignore`. Historical result links in this guide and `plans/`
require those local artifacts; a fresh clone must supply the dataset and
download the Qwen cache or regenerate the relevant results as described below.
The project has no selected top-level license; vendored TRM code
retains its own [MIT license](vendor/tiny_recursive_models/LICENSE).

## Experiments at a glance

| Experiment | Entry point | Evaluation images | Output parent |
| --- | --- | --- | --- |
| Qwen zero-shot | [Zero-shot notebook](notebooks/cs297-experiment1-em-local-before-resize-cap.ipynb) | All 200 images | `outputs/original_configuration_runs/` |
| Qwen few-shot | [Few-shot notebook](notebooks/cs297-experiment1-em-local-before-resize-cap-few-shot.ipynb) | 196 targets; four images of subject `0` serve as references | `outputs/original_configuration_few_shot_runs/` |
| Supervised TRM | [TRM notebook](notebooks/cs297-experiment1-em-local-trm.ipynb) or `python -m src.trm_training` | 40 held-out images after training on 140 and validating on 20 | `outputs/trm_supervised_runs/` |
| TRM stage/gradient diagnosis | `scripts/diagnose_trm.py` | Saved training and validation partitions | Explicit `--output` directory |
| TRM learnability controls | `scripts/diagnose_trm_controls.py` | Eight training images from two people | Explicit `--output` directory |

The synthetic notebook smoke run and optional overfit diagnostic are described
under [Validation](#validation). Saved notebooks under `outputs/` are execution
artifacts or historical evidence; use the three notebooks under `notebooks/` for
new real-data runs.

## 1. Prepare the environment

### Python and dependencies

Use Python 3.13 or 3.14. The dependency versions were originally recorded in the
Python 3.14 experiment environment; a fresh checkout can use Python 3.13:

```sh
python3.13 -m venv .venv
```

Install the dependencies for all active notebooks, TRM training, reporting,
Jupyter, and the complete test suite:

```sh
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` includes TRM's `einops` and `pydantic` dependencies. Its required
upstream reasoning code is already included in `vendor/tiny_recursive_models/`;
no separate upstream checkout or pretrained TRM download is needed.

Virtual environments are local to an interpreter installation. Before reusing
an existing environment, check `.venv/bin/python --version`. If its Python link
is broken after an interpreter upgrade, preserve the old environment elsewhere
and create a fresh `.venv` with one of the interpreters above, then reinstall
the requirements.

### Dataset

The source is Kaggle's
[Human Faces Augmentation dataset](https://www.kaggle.com/datasets/trainingdatapro/human-faces-augmentation-image-restoration).
After dependency installation, prepare it from the `experiment1/` directory:

```sh
.venv/bin/python -m src.dataset_setup --project-root .
```

This reuses complete local data without network access. Otherwise it downloads
the Kaggle dataset, locates its image folders inside any archive wrapper, checks
membership/pixels/sizes, and installs the prepared dataset. An incomplete existing
folder is preserved as `data/files.incomplete.<id>`. Failed downloads or validation
leave existing files intact. Kaggle source/version information is recorded in
`data/dataset_source.json`; cloud runs also record content hashes for all images.
No augmentations are generated: the downloaded dataset must contain all four
existing image groups.

For reproducible downloads, pass `--kaggle-version NUMBER` after selecting the
desired dataset version. Without it, KaggleHub resolves the latest version.
Private or consent-restricted access requires Kaggle authorization; use
`kagglehub.login()` or securely supply `KAGGLE_API_TOKEN`. GCP credentials do not
authenticate to Kaggle. See [KaggleHub authentication](https://github.com/Kaggle/kagglehub#authenticate).

To restore a prepared dataset from Cloud Storage instead:

```sh
.venv/bin/python -m src.dataset_setup --project-root . \
  --gcs-uri gs://YOUR_BUCKET/experiment1/inputs/data/files
```

The GCS prefix must contain `images/` and `augmentation/` (an enclosing folder is
also supported). Configure Application Default Credentials with permission to
read and list those objects. A selected GCS source does not fall back to Kaggle
on authentication or download errors.

Manual preparation remains supported with this layout:

```text
data/files/
├── images/                 # clean: severity 0
│   ├── 0.jpg
│   └── ... 49.jpg
└── augmentation/
    ├── 1/                 # weak: severity 1; 0.jpg ... 49.jpg
    ├── 2/                 # medium: severity 2; 0.jpg ... 49.jpg
    └── 3/                 # strong: severity 3; 0.jpg ... 49.jpg
```

The filename stem is the subject/person ID. The same 50 subjects must appear
once in each of the four folders: 200 images total, 50 per severity. The loaders
read the image folders directly; `data/augmentation.csv` and its spreadsheet
pointer are not execution inputs. The default TRM patch size requires every
image to be at least 128 pixels in each dimension.

Check the dataset before loading a model:

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
from src.image_quality_pipeline import build_dataset

data = build_dataset(Path.cwd() / "data/files")
assert len(data) == 200
assert data.groupby("severity").size().to_dict() == {"0": 50, "1": 50, "2": 50, "3": 50}
counts = data.groupby(["subject_id", "severity"]).size().unstack(fill_value=0)
assert len(counts) == 50 and counts.eq(1).all().all()
print(data.groupby(["augmentation", "severity"]).size())
PY
```

All experiments use these string labels:

| Severity | Meaning | Noise | Route |
| --- | --- | --- | --- |
| `"0"` | Clean | `"0"` | `"0"` — skip denoising |
| `"1"` | Weak degradation | `"1"` | `"0"` — skip denoising |
| `"2"` | Medium degradation | `"1"` | `"1"` — apply denoising |
| `"3"` | Strong degradation | `"1"` | `"1"` — apply denoising |

### Qwen model cache

Both Qwen notebooks load **`Qwen/Qwen3.5-0.8B`** with
`local_files_only=True` and set `HF_HOME` to this project's `hf_cache/`. A complete
local model/processor cache is therefore required before running either notebook.
If it is missing, populate it once with network access:

```sh
HF_HOME="$PWD/hf_cache" .venv/bin/hf download Qwen/Qwen3.5-0.8B
```

The active experiments use local inference and do not require an OpenAI client
or API key. `HF_TOKEN` is only needed if your model-download access requires it.
The notebooks check that the cached processor has resizing enabled and the
recovered defaults `longest_edge=16777216` and `shortest_edge=65536`; a different
cached configuration raises an error. Preserve the existing cache for comparison
with prior runs. TRM does not use this cache.

### Start Jupyter and select the environment

Register a named kernel inside the project environment and start Jupyter:

```sh
.venv/bin/python -m ipykernel install --sys-prefix --name experiment1 --display-name "CS297 experiment1 (.venv)"
.venv/bin/python -m jupyter lab
```

Select **CS297 experiment1 (.venv)** in Jupyter, or select the project
`.venv/bin/python` interpreter in VS Code. Notebook working-directory discovery
supports the project root and its `notebooks/` directory. To confirm the selected
kernel, run `import sys; print(sys.executable)` in a temporary cell.

## 2. Run Qwen zero-shot

1. Open [cs297-experiment1-em-local-before-resize-cap.ipynb](notebooks/cs297-experiment1-em-local-before-resize-cap.ipynb).
2. Select the project kernel, then **Restart Kernel → Run All**.
3. Check the dataset count (200), printed model device/dtype, and processor size.
4. Let the inference cell complete, then run the final evaluation/export cell
   if it has not already executed.

The notebook embeds its original prompt and evaluates all 200 images in one
pass, with one target per call, greedy generation (`do_sample=False`), and at most
128 new tokens. Images are converted to RGB and use the processor's normal
resizing/alignment. The filename's “before-resize-cap” means no custom pixel cap
or detail crop is applied. The notebook chooses MPS when available, otherwise
CPU; its current code does not select CUDA automatically.

Each setup creates a UTC timestamp/UUID session. Predictions and reports go to:

```text
outputs/original_configuration_runs/<session>/experiment_001/run_001/
```

For execution from a terminal instead of the UI, after registering the kernel:

```sh
.venv/bin/python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=experiment1 \
  --ExecutePreprocessor.timeout=-1 \
  --output zero_shot.executed.ipynb --output-dir /tmp \
  notebooks/cs297-experiment1-em-local-before-resize-cap.ipynb
```

This executes the full real-data experiment. The executed notebook is saved to
`/tmp/zero_shot.executed.ipynb`; prediction artifacts still use the session
directory above. Choose a different executed-notebook filename to retain
multiple copies. `timeout=-1` allows long inference cells to finish.

## 3. Run Qwen few-shot

1. Open [cs297-experiment1-em-local-before-resize-cap-few-shot.ipynb](notebooks/cs297-experiment1-em-local-before-resize-cap-few-shot.ipynb).
2. Keep `EXAMPLE_SUBJECT = "0"` for the existing experiment. Choose any alternative
   reference subject before inspecting predictions.
3. Select the project kernel, then **Restart Kernel → Run All**.
4. Confirm the reference table contains four labeled examples and the evaluation
   set contains 196 images (49 per severity).
5. Let inference and the final evaluation/export cell complete.

Every target conversation contains four reference-image/user turns with their
correct assistant JSON answers, followed by the target image. The shared runner
appends the few-shot guidance to the original prompt. All four images belonging
to the reference subject are excluded from evaluation. Model, image processing,
single-target batching, and generation settings match the zero-shot notebook.
Five images per conversation increase memory use and runtime.

Results go to:

```text
outputs/original_configuration_few_shot_runs/<session>/experiment_001/run_001/
```

Terminal alternative:

```sh
.venv/bin/python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=experiment1 \
  --ExecutePreprocessor.timeout=-1 \
  --output few_shot.executed.ipynb --output-dir /tmp \
  notebooks/cs297-experiment1-em-local-before-resize-cap-few-shot.ipynb
```

For a matched comparison, restrict zero-shot predictions to the same 196 target
images. The default zero-shot aggregate includes the reference subject's four
images, so its 200-image score uses a different evaluation set.

## 4. Run supervised TRM

### Train and evaluate from the notebook

1. Install `requirements.txt` and open
   [cs297-experiment1-em-local-trm.ipynb](notebooks/cs297-experiment1-em-local-trm.ipynb).
2. Leave `TRAIN_MODEL=True` (the default) and use
   [configs/trm_experiment1.json](configs/trm_experiment1.json).
3. Leave the optional `TRM_*` environment overrides unset for ordinary runs.
4. Select the project kernel, then **Restart Kernel → Run All**.
5. Inspect the split, patch preview, training history, held-out predictions,
   and final CSV report.

The default seed is 42. All four variants of a person remain together: 35 people
for training (140 images), five for validation (20), and ten for testing (40).
The model jointly trains a small CNN and recursive reasoning blocks on nine
native-scale 128×128 RGB patches. Defaults include width 128, two reasoning
layers, `H_cycles=2`, `L_cycles=2`, and four outer refinement steps.

Training uses batch size 8, AdamW with learning rate `0.0003`, at most 100 epochs,
and patience 15. The earliest best validation-loss checkpoint is selected and
evaluated on the held-out test partition. `device="auto"` chooses CUDA, then MPS,
then CPU. See [PIPELINE.md](PIPELINE.md) for architecture details and
[plans/04-validation-results.md](plans/04-validation-results.md) for the initial
execution record. The initial configuration did not learn useful image
discrimination; reproducing a successful execution does not imply strong scores.

### Train and evaluate from the terminal

```sh
OMP_NUM_THREADS=2 .venv/bin/python -m src.trm_training \
  --config configs/trm_experiment1.json \
  --project-root . \
  --session outputs/trm_supervised_runs/my_new_session \
  --evaluate
```

`--evaluate` runs the held-out pass and exports CSVs after training. Omit it for
training only. To change hyperparameters or choose `cpu`, `mps`, or `cuda`, copy
the JSON configuration, edit the copy, and pass it with `--config`. An explicitly
requested unavailable device raises an error. Training resume is not implemented;
use a fresh session path rather than an existing training session.

The notebook generates a timestamp/UUID session automatically; the terminal
command uses the session name you supply. Both produce:

```text
outputs/trm_supervised_runs/<session>/
├── config.json
├── split.json
├── training_manifest.json
├── training_history.csv
├── checkpoints/best.pt
├── evaluation_origin.json             # after held-out evaluation
└── experiment_001/run_001/            # predictions, metrics, and CSVs
```

The checkpoint contains CPU-portable weights, configuration, and split metadata.
Interrupted or failed training preserves the best completed checkpoint, if any,
and records its status.

### Evaluate an existing checkpoint without training

In the TRM notebook's setup cell, replace the `TRAIN_MODEL` and `CHECKPOINT_PATH`
assignments with:

```python
TRAIN_MODEL = False
CHECKPOINT_PATH = PROJECT_PATH / "outputs/trm_supervised_runs/20261024_implementation_validation/checkpoints/best.pt"
```

Then execute the notebook from the setup cell onward. If using **Restart Kernel
→ Run All**, save those edited assignments first. A new evaluation session is
created using the checkpoint's original configuration and person split; image
hashes and membership are verified. Change the checkpoint path to evaluate
another saved model.

Alternatively, run this from the project root:

```sh
OMP_NUM_THREADS=2 .venv/bin/python - <<'PY'
from pathlib import Path
from src.image_quality_pipeline import build_dataset
from src.trm_adapter import evaluate_checkpoint

root = Path.cwd()
checkpoint = root / "outputs/trm_supervised_runs/20261024_implementation_validation/checkpoints/best.pt"
session = root / "outputs/trm_supervised_runs/my_checkpoint_evaluation"
if session.exists():
    raise FileExistsError(f"Choose a new evaluation session: {session}")
run = evaluate_checkpoint(checkpoint, build_dataset(root / "data/files"), root, session)
print("Saved held-out predictions and CSVs:", run)
PY
```

Optional notebook environment overrides are `TRM_CONFIG_PATH`,
`TRM_DATA_PROJECT_ROOT` (a root containing `data/files/`), `TRM_OUTPUT_ROOT`,
`TRM_TRAIN_MODEL` (`true`/`false`), and `TRM_CHECKPOINT_PATH`. The smoke script
uses them to substitute synthetic inputs and a tiny configuration.

## 5. Run TRM diagnostic experiments

Install `requirements.txt` first. These scripts run on CPU with two Torch
threads and require **new output directories**. Both default to this saved
checkpoint; use `--checkpoint PATH` to choose another compatible checkpoint:

```text
outputs/trm_supervised_runs/20261024_implementation_validation/checkpoints/best.pt
```

### Stage, feature, probability, and gradient trace

```sh
OMP_NUM_THREADS=2 .venv/bin/python scripts/diagnose_trm.py \
  --output outputs/trm_diagnostics/my_new_trace
```

This compares seed-matched initial and trained models on the saved training and
validation partitions, checks image hashes, and traces information through the
encoder and recurrence. It writes `stages.csv`, `probabilities.csv`,
`gradients.json`, `truncation_check.json`, `centroid_probes.json`, `features.pt`,
and `manifest.json`. It does not read test image pixels.

### Learnability and architecture controls

Run the default five controls, each for 400 optimizer updates:

```sh
OMP_NUM_THREADS=2 .venv/bin/python scripts/diagnose_trm_controls.py \
  --output outputs/trm_diagnostics/my_new_controls
```

The defaults are `original`, `shallow`, `normalized_frozen_tokens`, `cnn_mean`,
and `cnn_mean_std`. To run the separate recursion and matched-scaling controls:

```sh
OMP_NUM_THREADS=2 .venv/bin/python scripts/diagnose_trm_controls.py \
  --output outputs/trm_diagnostics/my_new_recursion \
  --variants h1 l1 k1

OMP_NUM_THREADS=2 .venv/bin/python scripts/diagnose_trm_controls.py \
  --output outputs/trm_diagnostics/my_new_scaling \
  --variants raw_frozen_tokens normalized_frozen_tokens
```

Every available control can also be executed in one invocation:

```sh
OMP_NUM_THREADS=2 .venv/bin/python scripts/diagnose_trm_controls.py \
  --output outputs/trm_diagnostics/my_all_controls --updates 400 \
  --variants original shallow h1 l1 k1 raw_frozen_tokens normalized_frozen_tokens cnn_mean cnn_mean_std
```

| Variant | Change from the original control |
| --- | --- |
| `original` | Joint CNN + TRM, original recursion budget |
| `shallow` | Set H, L, and outer refinement count K to 1 |
| `h1` | Set only H to 1 |
| `l1` | Set only L to 1 |
| `k1` | Set only K to 1 |
| `raw_frozen_tokens` | Freeze initial encoder/readout; train TRM on raw tokens |
| `normalized_frozen_tokens` | Same frozen encoder/readout; standardize image tokens using the selected training subset |
| `cnn_mean` | CNN spatial means and a direct classifier, without TRM |
| `cnn_mean_std` | CNN spatial means and standard deviations with a direct classifier |

`--updates N` changes the optimizer-update budget. Controls start from a
seed-matched initialization, using the checkpoint for configuration and split,
and repeatedly fit eight balanced training images from two people. Results
describe subset learnability, not held-out performance. Each output directory
contains `manifest.json`, `history.csv`, `summary.json`, and a separate
`<variant>.pt` state dictionary. These diagnostic state dictionaries are not
production `best.pt` checkpoints for `evaluate_checkpoint`.

Interpretation and prior measured results are in
[plans/05-learning-diagnosis.md](plans/05-learning-diagnosis.md).

## 6. Review results without rerunning inference

Every experiment uses a prediction folder ending in `experiment_001/run_001/`:

| File | Contents |
| --- | --- |
| `manifest.json` | Prompt/configuration, mode, progress, status, and runtime |
| `labels.json` | Ground-truth labels keyed by image path |
| `batch_*.json` | Per-image raw response, parsed prediction or error, and elapsed time |
| `metrics.json` | Aggregate noise, severity, and route metrics |
| `predictions.csv` | Individual predictions with ground truth and correctness |
| `failures.csv` | Invalid or failed predictions |
| `evaluation_summary.csv` | Task-level metrics and correct/attempted/planned ratios |
| `coverage_by_severity.csv` | Attempted, valid, failed, and planned counts by severity |

The notebooks' final **Evaluate and export saved predictions** / **Offline review
and CSV export** section can run independently. Start a fresh kernel, define an
absolute path first, then execute only the final reporting cell:

```python
from pathlib import Path
RUN_DIR = Path("/absolute/path/to/experiment_001/run_001")
```

No neural model or checkpoint is loaded during this review. From the terminal,
recompute JSON metrics for any of the three experiment types with:

```sh
.venv/bin/python -m src.evaluation \
  outputs/trm_supervised_runs/20261024_implementation_validation/experiment_001/run_001
```

To export all four CSVs for any saved run in the shared format, replace the path
in this command as needed:

```sh
.venv/bin/python - <<'PY'
from src.trm_reporting import export_saved_run

report = export_saved_run(
    "outputs/trm_supervised_runs/20261024_implementation_validation/experiment_001/run_001"
)
print("Complete:", report["complete"])
print(report["summary"])
PY
```

Accuracy, precision, recall, F1, confusion matrices, and severity MAE score valid
predictions. Coverage and failures are separate. Correct/attempted counts failed
attempts as unsuccessful, and correct/planned also exposes unattempted images.
Incomplete runs remain partial results. Invalid JSON or inconsistent labels are
failures; the Qwen notebooks do not repair or retry responses.

Completed per-image checkpoints survive interruption. For new Qwen inference,
rerun setup to generate a new session; rerunning only its inference cell against
the existing `RUN_DIR` raises an error. Review the interrupted run offline.
For TRM retraining, choose a new session; evaluate any preserved best checkpoint
separately if desired.

### Historical and generated notebooks under `outputs/`

`outputs/recovered_history/` holds the recovered zero-shot notebook, pipeline
snapshot, and historical metrics. Open its notebook to inspect saved outputs;
its **Original run — offline evaluation review** code cells can be run on their
own to redisplay embedded historical tables. Use the active zero-shot notebook
for a fresh execution of that experiment. The later historical cells retain
old setup and demonstration code and are provenance rather than an additional
active experiment.

Notebooks under `outputs/trm_validation/` are saved synthetic executions of the
TRM notebook. Regenerate them with the smoke command below instead of treating
their fixture paths as a real-data experiment configuration.

## 7. Validation

### Shared-code tests

After installing `requirements.txt`:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The default suite uses synthetic fixtures and needs no model downloads. Device
smoke checks run when the relevant device is available; the longer overfit
diagnostic is opt-in.

### Synthetic notebook smoke run

To validate the new cloud notebooks without network access or real data:

```sh
.venv/bin/python scripts/validate_colab_notebooks.py \
  --output outputs/colab_validation/my_new_smoke
```

This executes fresh kernels for real tiny TRM training, checkpoint-only evaluation,
and both Qwen notebook workflows using a fixture adapter. It checks reports,
few-shot exclusions, and local artifact provenance. It does not download Qwen or
validate a live GPU/GCP environment. Cloud Storage calls are covered separately
with mocked clients in the unit suite.

For the original local TRM notebook:

```sh
.venv/bin/python scripts/validate_trm_notebook.py \
  --output outputs/trm_validation/my_new_smoke
```

This creates four synthetic people with four 16×16 images each, trains a tiny
CPU model for two epochs with a 2/1/1 person split, runs checkpoint-only
evaluation, and checks model-free offline export. It uses fresh `python3`
Jupyter kernels from the project environment and needs permission to open local
kernel sockets. Launch it with `.venv/bin/python` so the environment is placed
first on the kernel process's `PATH`.

Its new output directory contains synthetic fixtures, `smoke_config.json`,
session artifacts, `summary.json`, and three executed notebooks:
`training_from_root.ipynb`, `checkpoint_from_notebooks.ipynb`, and
`offline_only.ipynb`. The real dataset is not used.

### Optional synthetic four-class overfit diagnostic

```sh
TRM_OVERFIT_DIAGNOSTIC=1 OMP_NUM_THREADS=2 .venv/bin/python \
  -m unittest discover -s tests -p test_trm_training.py -k tiny_four_class -v
```

This explicitly enables the longer synthetic learnability check. For real-image
subset overfitting controls, use `diagnose_trm_controls.py` above.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `ModuleNotFoundError` | Select the project `.venv` kernel/interpreter and install the relevant requirements files. |
| Notebook cannot locate the project | Start Jupyter from `experiment1`; keep the notebook working directory at the root or `notebooks/`. |
| Qwen cannot find cached files | Populate `hf_cache/` using the command in setup; active notebooks intentionally load only local files. |
| Processor defaults differ | Check that the expected cached processor/revision is present; the notebook checks preserve the recovered experiment settings. |
| Dataset count or subject assertions fail | Restore all four severity folders and matching IDs `0`–`49`; do not include extra images. |
| TRM reports an image smaller than `patch_size` | Restore the original images or use a separate configuration with a suitable patch size. |
| Checkpoint image-hash/split verification fails | Use the checkpoint's original dataset content and person membership. |
| Existing run/output directory error | Choose a fresh session or diagnostic output path; use offline review for existing results. |
| Few-shot inference exceeds available memory | Run it in a fresh kernel with other model sessions closed; the five-image conversation is part of this experiment's setup. |
| Smoke run cannot start a kernel | Check local socket permission and the `.venv` `python3` kernelspec; use the project interpreter. |

For workflow context, see [PIPELINE.md](PIPELINE.md). The TRM specifications and
execution/diagnosis records are indexed in [plans/README.md](plans/README.md).
