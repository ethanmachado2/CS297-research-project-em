"""Build the three cleared, standalone cloud entry notebooks from shared cells."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

PROJECT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT / "notebooks/colab_enterprise"


def markdown(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


CONFIG = '''
import os
from pathlib import Path

# Edit these settings before Run All. For an uploaded checkout, use its root.
REPO_URL = "https://github.com/ethanmachado2/CS297-research-project-em.git"
REPO_REF = os.environ.get("CS297_REPO_REF", "main")  # Prefer a commit SHA for a recorded experiment.
REPO_DIR = Path(os.environ.get("CS297_REPO_DIR", str(Path.home() / "cs297-research"))).expanduser()
INSTALL_DEPENDENCIES = os.environ.get("CS297_INSTALL_DEPENDENCIES", "true").lower() == "true"
DEVICE = os.environ.get("CS297_DEVICE", "cuda")

# Enable end-user credentials in the runtime. No service-account keys are needed.
# This is an existing bucket/prefix you can write to, not a bucket to create.
GCS_OUTPUT_URI = os.environ.get("CS297_GCS_OUTPUT_URI", "")  # e.g. gs://my-bucket/cs297/runs
REQUIRE_GCS_OUTPUT = os.environ.get("CS297_REQUIRE_GCS_OUTPUT", "true").lower() == "true"
GCS_INPUT_URI = os.environ.get("CS297_GCS_INPUT_URI", "")  # Optional: prepared images/ + augmentation/ prefix.
KAGGLE_HANDLE = os.environ.get("CS297_KAGGLE_HANDLE", "trainingdatapro/human-faces-augmentation-image-restoration")
KAGGLE_VERSION = os.environ.get("CS297_KAGGLE_VERSION") or None  # Set a version to reproduce a download.
EXPECTED_SUBJECTS = int(os.environ.get("CS297_EXPECTED_SUBJECTS", "50"))
MIN_IMAGE_SIZE = int(os.environ.get("CS297_MIN_IMAGE_SIZE", "128"))
'''

BOOTSTRAP = '''
import subprocess
import sys

if not REPO_DIR.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", REPO_REF], check=True)
# Existing checkouts are reused without fetching, checking out, or overwriting edits.
PROJECT_PATH = REPO_DIR.resolve() / "experiment1"
installer = PROJECT_PATH / "scripts/install_cloud_dependencies.py"
if not installer.is_file():
    raise RuntimeError("This checkout needs the cloud-porting files. Push them to your repository "
                       "before cloning, or set REPO_DIR to an uploaded checkout containing them.")
os.chdir(PROJECT_PATH)
if INSTALL_DEPENDENCIES:
    subprocess.run([sys.executable, str(installer)], check=True)
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))
os.environ["HF_HOME"] = str(PROJECT_PATH / "hf_cache")
print("Project:", PROJECT_PATH)
if (REPO_DIR / ".git").exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], check=True)
'''

PREFLIGHT = '''
import torch
from src.cloud_runtime import parse_gcs_uri

if REQUIRE_GCS_OUTPUT and not GCS_OUTPUT_URI:
    raise ValueError("Set GCS_OUTPUT_URI in the settings cell before running. "
                     "For an intentional local-only run, set REQUIRE_GCS_OUTPUT=False.")
if GCS_OUTPUT_URI:
    parse_gcs_uri(GCS_OUTPUT_URI)
if DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Connect to a GPU runtime, then restart this notebook.")
print("Python:", sys.version)
print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("Device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE)
'''

DATASET = '''
from src.dataset_setup import ensure_dataset
from src.image_quality_pipeline import build_dataset

# Existing complete data is reused without network access. Otherwise use GCS
# when configured, or download Kaggle data and normalize its archive layout.
DATA_ROOT, dataset_manifest = ensure_dataset(
    PROJECT_PATH, kaggle_handle=KAGGLE_HANDLE, kaggle_version=KAGGLE_VERSION,
    gcs_uri=GCS_INPUT_URI or None, expected_subjects=EXPECTED_SUBJECTS,
    min_image_size=MIN_IMAGE_SIZE, return_manifest=True,
)
df_experiment = build_dataset(DATA_ROOT)
display(df_experiment.groupby(["augmentation", "severity"]).size().rename("images").to_frame())
print("Dataset digest:", dataset_manifest["content_digest"])
'''

SESSION = '''
from datetime import datetime, timezone
from uuid import uuid4
from src.cloud_runtime import ArtifactSync, write_run_provenance

SESSION_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
SESSION_DIR = PROJECT_PATH / "outputs/colab_enterprise" / WORKFLOW / SESSION_ID
GCS_SESSION_URI = (GCS_OUTPUT_URI.rstrip("/") + "/" + WORKFLOW + "/" + SESSION_ID
                   if GCS_OUTPUT_URI else None)
settings = {"workflow": WORKFLOW, "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
            "cuda": torch.version.cuda}
write_run_provenance(SESSION_DIR, PROJECT_PATH, dataset_manifest, settings)
print("Local session:", SESSION_DIR)
print("Cloud session:", GCS_SESSION_URI or "LOCAL ONLY — copy artifacts before runtime deletion")
'''

REPORT = '''
from src.trm_reporting import export_saved_run

report = export_saved_run(RUN_DIR)
display(report["summary"])
display(report["coverage"])
display(report["predictions"].head(20))
print("Complete:", report["complete"], "Failures:", len(report["failures"]))
artifact_sync.sync()  # Include any reports regenerated in this cell; safe to retry after an upload failure.
print("Saved:", GCS_SESSION_URI or SESSION_DIR)
'''


def build(workflow):
    trm = workflow == "trm_supervised"
    few = workflow == "qwen_few_shot"
    title = "TRM supervised training and evaluation" if trm else "Qwen " + ("few-shot" if few else "zero-shot") + " inference"
    cells = [markdown(f'''
        # CS297 — {title} on Colab Enterprise

        This notebook contains the setup and execution steps for a fresh runtime.
        It clones the repository, installs dependencies, prepares the Kaggle/GCS
        dataset, runs the experiment, exports CSVs, and uploads artifacts to GCS.
        It also runs in Vertex/Agent Platform Workbench with the same settings.

        **Before Run All:** connect to a PyTorch GPU runtime, set `GCS_OUTPUT_URI`,
        and ensure this repository revision contains the cloud-porting files.
        Your runtime identity needs input read and output write permissions.
        GitHub, PyPI, Kaggle, and Hugging Face downloads require network access.
        For a private/consent-restricted Kaggle dataset, accept its terms and set
        `KAGGLE_API_TOKEN` securely before the data cell (or use `kagglehub.login()`).
        Never put credentials in notebook source or outputs.

        Existing local notebooks and historical experiments remain separate.
        ''')]
    config = CONFIG + f'\nWORKFLOW = "{workflow}"\n'
    if trm:
        config += '''
TRAIN_MODEL = os.environ.get("CS297_TRAIN_MODEL", "true").lower() == "true"
CHECKPOINT_PATH = os.environ.get("CS297_CHECKPOINT_PATH", "")  # local path or gs://.../best.pt for evaluation only
TRM_CONFIG_PATH = os.environ.get("CS297_TRM_CONFIG_PATH", "")  # optional alternate JSON configuration
'''
    else:
        config += '''
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = os.environ.get("CS297_MODEL_REVISION") or None  # commit SHA; resolved SHA is saved
EXAMPLE_SUBJECT = "0"
'''
    cells += [markdown("## 1. Configure this run"), code(config),
              markdown("## 2. Clone code and install dependencies\n\nThe installer reads the single `requirements.txt`, retaining managed Torch and Jupyter packages. If pip asks for a kernel restart, restart and rerun these cells. Existing checkouts are reused; update them explicitly when you want new code."),
              code(BOOTSTRAP), markdown("## 3. Check GPU and output settings"), code(PREFLIGHT),
              markdown("## 4. Prepare and validate data\n\nThe first download may take time. Existing data is validated by subject membership, image decoding, size, and hashes. Kaggle/GCS failures stop execution. An incomplete local dataset is preserved before replacement."),
              code(DATASET), markdown("## 5. Create an isolated run"), code(SESSION)]
    if trm:
        cells += [markdown("## 6. Train or evaluate a saved checkpoint\n\nThe default configuration uses 35/5/10 subjects for training/validation/test. Keep subjects together and tune using validation results. Checkpoint-only mode verifies original image hashes and split membership. Training resume is not implemented; use a new session for retraining."), code('''
            import json
            from src.trm_training import train_experiment
            from src.trm_adapter import evaluate_checkpoint
            from src.cloud_runtime import download_gcs_file

            config_path = Path(TRM_CONFIG_PATH) if TRM_CONFIG_PATH else PROJECT_PATH / "configs/trm_experiment1.json"
            config = json.loads(config_path.read_text())
            config["training"]["device"] = DEVICE
            if not TRAIN_MODEL and not CHECKPOINT_PATH:
                raise ValueError("Set CHECKPOINT_PATH when TRAIN_MODEL=False")
            # Upload changed artifacts every 60 seconds and again on Python errors/interruption.
            with ArtifactSync(SESSION_DIR, GCS_SESSION_URI) as artifact_sync:
                if TRAIN_MODEL:
                    checkpoint = train_experiment(df_experiment, PROJECT_PATH, SESSION_DIR, config)
                elif CHECKPOINT_PATH.startswith("gs://"):
                    checkpoint = download_gcs_file(CHECKPOINT_PATH, SESSION_DIR / "input_checkpoint/best.pt")
                else:
                    checkpoint = Path(CHECKPOINT_PATH).expanduser().resolve()
                RUN_DIR = evaluate_checkpoint(checkpoint, df_experiment, PROJECT_PATH, SESSION_DIR, device=DEVICE)
                display(json.loads((SESSION_DIR / "config.json").read_text()))
            ''')]
    else:
        cells += [markdown("## 6. Download the model and run inference\n\nPreserves the original prompt, float32 precision, processor defaults, batch size 1, greedy decoding, and 128-token limit. Few-shot mode excludes the reference subject from evaluation. First model download requires Hugging Face access; cached snapshots are reused."), code('''
            import json
            from src.qwen_adapter import QwenImageModel
            from src.image_quality_pipeline import prepare_few_shot_split, run_inference
            from src.trm_reporting import export_saved_run

            mode = "few_shot" if WORKFLOW == "qwen_few_shot" else "zero_shot"
            prompt = (PROJECT_PATH / "configs/qwen_original_prompt.txt").read_text()
            examples = ()
            evaluation = df_experiment
            if mode == "few_shot":
                evaluation, examples = prepare_few_shot_split(df_experiment, EXAMPLE_SUBJECT)
            print("Evaluation images:", len(evaluation), "Reference images:", len(examples))
            RUN_DIR = SESSION_DIR / "experiment_001/run_001"
            with ArtifactSync(SESSION_DIR, GCS_SESSION_URI) as artifact_sync:
                adapter = QwenImageModel.from_pretrained(
                    MODEL_NAME, cache_dir=str(PROJECT_PATH / "hf_cache/hub"), revision=MODEL_REVISION,
                    device=DEVICE, mode=mode, example_subject=EXAMPLE_SUBJECT,
                )
                (SESSION_DIR / "model_config.json").write_text(json.dumps(adapter.inference_config, indent=2) + "\\n")
                results = run_inference(
                    model=adapter, image_paths=evaluation.file_path.tolist(), prompt=prompt,
                    batch_size=1, output_dir=SESSION_DIR, experiment_number=1, run_number=1,
                    show_progress=True, labels=evaluation, model_name=MODEL_NAME,
                    inference_mode=mode, examples=examples, response_format="json",
                )
                export_saved_run(RUN_DIR)
            ''')]
    cells += [markdown("## 7. Compare results and verify storage\n\nReports and provenance are kept with the session. Periodic uploads reduce loss from runtime deletion; a hard VM termination can still lose changes since the last upload. The model cache is local and can be downloaded again. Set a model/dataset version and repository commit for repeatable runs."), code(REPORT)]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": workflow + ".ipynb"}})
    # Stable IDs keep regeneration diffs readable.
    for index, cell in enumerate(notebook.cells):
        cell.id = f"{workflow}-{index:02d}"
    nbf.validate(notebook)
    return notebook


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for workflow in ("trm_supervised", "qwen_zero_shot", "qwen_few_shot"):
        nbf.write(build(workflow), DESTINATION / f"{workflow}.ipynb")


if __name__ == "__main__":
    main()
