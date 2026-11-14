"""Execute cloud notebooks on synthetic CPU data without downloads or cloud calls.

TRM uses real tiny training/checkpoint evaluation. Qwen uses a fixture adapter,
verifying orchestration, reference exclusions, reports, and artifact lifecycle
without downloading model weights. This does not certify a CUDA/GCP environment.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests"))
from trm_fixtures import make_dataset, tiny_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # A standalone checkout skeleton isolates synthetic data from the real data.
    checkout = output / "checkout"
    fixture_project = checkout / "experiment1"
    fixture_project.mkdir(parents=True)
    for name in ("src", "vendor", "configs", "scripts"):
        (fixture_project / name).symlink_to(PROJECT / name, target_is_directory=True)
    make_dataset(fixture_project)
    config_path = output / "tiny_config.json"
    config_path.write_text(json.dumps(tiny_config()))
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
               OMP_NUM_THREADS="2", CS297_REPO_DIR=str(checkout), CS297_INSTALL_DEPENDENCIES="false",
               CS297_DEVICE="cpu", CS297_REQUIRE_GCS_OUTPUT="false", CS297_GCS_OUTPUT_URI="",
               CS297_GCS_INPUT_URI="", CS297_EXPECTED_SUBJECTS="4", CS297_MIN_IMAGE_SIZE="16",
               CS297_TRM_CONFIG_PATH=str(config_path), CS297_TRAIN_MODEL="true", CS297_CHECKPOINT_PATH="")
    results = []

    def execute(workflow, name, *, checkpoint=None):
        notebook = deepcopy(nbformat.read(PROJECT / f"notebooks/colab_enterprise/{workflow}.ipynb", as_version=4))
        run_env = dict(env)
        if checkpoint:
            run_env.update(CS297_TRAIN_MODEL="false", CS297_CHECKPOINT_PATH=str(checkpoint))
        if workflow.startswith("qwen"):
            # Inject just before experiment execution, after code/data setup.
            index = next(i for i, cell in enumerate(notebook.cells)
                         if cell.cell_type == "code" and "QwenImageModel.from_pretrained(" in cell.source)
            notebook.cells.insert(index, nbformat.v4.new_code_cell('''
from src.qwen_adapter import QwenImageModel
class FixtureQwen:
    supports_few_shot = True
    inference_config = {"fixture_adapter": True, "device": "cpu", "dtype": "float32"}
    def generate(self, image_path, prompt, *, examples=()):
        return '{"noise":"0","severity":"0","route":"0"}'
QwenImageModel.from_pretrained = classmethod(lambda cls, *args, **kwargs: FixtureQwen())
'''))
        NotebookClient(notebook, timeout=120, kernel_name="python3",
                       resources={"metadata": {"path": str(output)}}).execute(env=run_env)
        nbformat.write(notebook, output / f"{name}.ipynb")
        parent = fixture_project / "outputs/colab_enterprise" / workflow
        session = max(parent.iterdir(), key=lambda p: p.stat().st_mtime_ns)
        manifest = json.loads((session / "experiment_001/run_001/manifest.json").read_text())
        planned = 4 if workflow == "trm_supervised" else (12 if workflow == "qwen_few_shot" else 16)
        assert manifest["status"] == "completed" and manifest["completed_images"] == planned
        assert json.loads((session / "cloud_status.json").read_text())["status"] == "completed"
        for name in ("predictions.csv", "failures.csv", "evaluation_summary.csv", "coverage_by_severity.csv"):
            assert (session / "experiment_001/run_001" / name).is_file()
        for name in ("environment.json", "dataset_manifest.json"):
            assert (session / name).is_file()
        results.append({"workflow": workflow, "images": planned, "session": str(session)})
        return session

    training = execute("trm_supervised", "trm_training")
    execute("trm_supervised", "trm_checkpoint_only", checkpoint=training / "checkpoints/best.pt")
    execute("qwen_zero_shot", "qwen_zero_shot_fixture")
    execute("qwen_few_shot", "qwen_few_shot_fixture")
    (output / "summary.json").write_text(json.dumps({"status": "passed", "runs": results}, indent=2))
    print(json.dumps({"status": "passed", "runs": results}, indent=2))


if __name__ == "__main__":
    main()
