# Colab Enterprise and GCP Workbench

These notebooks are independent cloud entry points. Each includes settings,
repository/dependency bootstrap, dataset preparation, experiment execution, CSV
reports, and periodic/final Cloud Storage uploads. They reuse the tested Python
modules in the repository. Importing an IPYNB alone is sufficient to start the
bootstrap; it then clones the rest of the source.

| Notebook | Execution |
| --- | --- |
| [trm_supervised.ipynb](trm_supervised.ipynb) | Train TRM and evaluate the best validation checkpoint; optionally evaluate a saved checkpoint |
| [qwen_zero_shot.ipynb](qwen_zero_shot.ipynb) | Original Qwen prompt and processor settings; all 200 images |
| [qwen_few_shot.ipynb](qwen_few_shot.ipynb) | Four labeled reference images; 196 evaluation targets |

The original notebooks under `../` remain local entry points.

## 1. Prepare GCP

1. Select a project with billing, Colab Enterprise access, and sufficient GPU quota.
   Enable the APIs required by the console setup flow.
2. Create a PyTorch GPU runtime/template in the notebook's region and connect it.
   Start with one GPU and measure memory before choosing a larger machine.
3. Create a Cloud Storage bucket, preferably in the same region, and choose a run
   prefix such as `gs://YOUR_BUCKET/cs297/runs`. The notebooks do not create buckets,
   modify IAM, or deploy endpoints.
4. Enable end-user credentials. The runtime identity needs permission to write
   output objects and, for GCS inputs/checkpoints, to read and list input objects.
   Workbench uses its configured Application Default Credentials identity instead.
5. Allow access to GitHub/PyPI/Kaggle/Hugging Face, or pre-stage the corresponding
   source, wheels, data, and model snapshots for a private-network environment.

See Google's [notebook import guide](https://docs.cloud.google.com/colab/docs/create-console-quickstart),
[runtime template setup](https://docs.cloud.google.com/colab/docs/create-runtime-template),
and [credentials guide](https://docs.cloud.google.com/colab/docs/run-code-adc).

## 2. Import and configure

Commit/push these cloud-porting files to your repository before cloning from a
fresh runtime. Import one notebook into Colab Enterprise. Alternatively, open it
in Workbench JupyterLab with a checkout already available.

Edit the first settings cell:

- `REPO_URL`, `REPO_REF`, and `REPO_DIR`: the source checkout. Use a commit SHA for
  repeatable experiments. `REPO_REF` is applied only to newly cloned checkouts;
  an existing checkout is reused without overwriting edits. Update it explicitly
  when you want newer code. In Workbench, point `REPO_DIR` to the repository root.
- `GCS_OUTPUT_URI`: an existing writable `gs://bucket/prefix`. This is required by
  default to prevent a cloud run from silently leaving all results on the VM.
- `GCS_INPUT_URI`: optional prefix containing the prepared dataset. Otherwise the
  Kaggle source below is used when local data is incomplete.
- `KAGGLE_VERSION`: choose a version for repeatability. Leaving it unset downloads
  the latest version on first setup. The resolved cache version is recorded when
  available, and all runs record per-image hashes.
- `DEVICE`: defaults to `cuda`, and fails clearly if CUDA is unavailable.
- TRM: `TRM_CONFIG_PATH` selects an alternate configuration. For checkpoint-only
  evaluation, set `TRAIN_MODEL=False` and `CHECKPOINT_PATH` to a local file or
  `gs://.../best.pt`.
- Qwen: `MODEL_REVISION` can pin a Hugging Face commit. The resolved snapshot SHA
  is saved. `EXAMPLE_SUBJECT` defaults to `"0"` for few-shot inference.

Then **Run All**. Installation runs through the current kernel interpreter.
If pip requests a kernel restart, restart and rerun the setup cells before
continuing. Set `INSTALL_DEPENDENCIES=False` to reuse an already prepared environment.
The installer reads `experiment1/requirements.txt`, excludes managed Jupyter/Torch
entries, and constrains installed Torch/Torchvision versions during resolution.
There is no second TRM or cloud requirements list.

The local pins were recorded on Python 3.14; Google currently documents Python
3.12 as the Colab default. The installer preserves the CUDA build, but the selected
runtime still needs a compatible set of wheels. A resolver failure is an
environment compatibility issue to resolve before training; do not silently drop
dependency errors. Target-runtime installation/CUDA execution require validation
on GCP, beyond the repository's synthetic CPU checks.

## 3. Dataset preparation

Source: [Human Faces Augmentation](https://www.kaggle.com/datasets/trainingdatapro/human-faces-augmentation-image-restoration)
(`trainingdatapro/human-faces-augmentation-image-restoration`).

`ensure_dataset` checks `experiment1/data/files/`. A complete local copy is reused
without downloading. It verifies 50 subjects, four severities per subject, 200
decodable images, and minimum dimensions of 128×128. It hashes image bytes and
does not resize or regenerate augmentations.

When data is missing or incomplete, GCS is used if configured; otherwise KaggleHub
downloads the chosen Kaggle version. Nested `files/` or `data/files/` archive
wrappers are located automatically. Data is staged and validated before it
replaces the target; an incomplete old folder is preserved under
`data/files.incomplete.<id>`. Download/validation errors stop the notebook before
model loading and leave the old dataset intact.

If Kaggle requires authentication/consent, accept the dataset terms in Kaggle and
use `kagglehub.login()` or supply `KAGGLE_API_TOKEN` through your runtime's secure
credential mechanism before the dataset cell. Never write a token into the
notebook. [KaggleHub authentication and downloads](https://github.com/Kaggle/kagglehub).

For local setup, run from `experiment1/`:

```sh
.venv/bin/python -m src.dataset_setup --project-root .
# Optionally add --kaggle-version NUMBER or --gcs-uri gs://BUCKET/INPUT_PREFIX.
```

To reuse the same prepared inputs across GCP runtimes, upload once:

```sh
gcloud storage rsync --recursive data/files gs://YOUR_BUCKET/cs297/inputs/data/files
```

Set `GCS_INPUT_URI` to that prefix. Existing valid local data is always reused;
changing the requested source does not replace it. Use a separate checkout/data
directory when intentionally switching dataset versions. The recorded content
digest identifies the bytes actually used, rather than merely the requested URL.

## 4. Outputs and interruption

Each run creates a fresh timestamp/UUID session, locally and in Cloud Storage:

```text
outputs/colab_enterprise/<workflow>/<session>/
gs://YOUR_BUCKET/cs297/runs/<workflow>/<session>/
  environment.json             # commit, working-tree status, packages, device, source
  dataset_manifest.json        # relative image paths, labels, content hashes
  cloud_status.json            # running/completed/failed/interrupted
  config.json / model_config.json
  checkpoints/best.pt          # TRM training
  split.json                   # TRM
  experiment_001/run_001/      # predictions, metrics, four CSV reports
```

Uploads run every 60 seconds, after experiment completion, and on Python
exceptions/interruptions. Only changed files are uploaded; temporary checkpoint
files are excluded. The first upload checks output access before training/model
loading. Upload failures are surfaced, and local artifacts remain available.
Retry `artifact_sync.sync()` after restoring access. The final report cell also
uploads regenerated reports.

A hard VM termination can lose changes since the last successful sync. TRM
checkpoints support inference/evaluation, but training resume is not implemented.
Do not use an existing session for retraining. Model downloads/cache are local;
they can be fetched again. Preserve datasets and results in GCS before deleting
runtimes. For an intentional local-only test, set `REQUIRE_GCS_OUTPUT=False` and
leave `GCS_OUTPUT_URI` empty.

Qwen settings match the original experiment: float32, RGB whole images, unchanged
processor defaults, greedy decoding, one target at a time, and 128 generated
tokens. Few-shot conversations have five images and require more memory. GPU
execution can introduce numerical differences from historical MPS/CPU runs.

## 5. Verify changes locally

From `experiment1/`:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_colab_notebooks.py --output /tmp/cs297-cloud-smoke
```

The smoke run uses fresh kernels and isolated synthetic data: real tiny TRM
training, checkpoint-only evaluation, and fixture Qwen inference in both modes.
It checks exclusions, reports, and provenance without downloads or GCP calls.
Unit tests mock Kaggle/GCS for preparation, path validation, and upload failures.
They do not certify real Kaggle archive contents, model downloads, package
installation on the target runtime, or CUDA execution.

Notebook sources are generated reproducibly with:

```sh
.venv/bin/python scripts/build_colab_notebooks.py
```

Edit that generator for shared notebook changes, regenerate, and commit the
cleared notebooks. All saved source notebooks have empty outputs and execution
counts, keeping credentials and generated artifacts out of Git.
