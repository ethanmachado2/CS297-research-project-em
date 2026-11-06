"""Execute training, checkpoint-only, and model-free notebook smoke runs.

Uses tiny synthetic files; never trains or evaluates the real dataset. Saved
executed notebooks and a JSON summary go under --output for inspection.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "notebooks/cs297-experiment1-em-local-trm.ipynb"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = output / "fixture"
    for severity in range(4):
        folder = fixture / "data/files" / ("images" if severity == 0 else f"augmentation/{severity}")
        folder.mkdir(parents=True)
        for person in range(4):
            pixels = np.random.default_rng(person + severity * 10).integers(0, 256, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{person}.png")
    config = json.loads((PROJECT / "configs/trm_experiment1.json").read_text())
    config["split"]["counts"] = [2, 1, 1]
    config["preprocessing"].update(patch_size=8, grid_size=1)
    config["model"].update(hidden_size=16, num_heads=2, H_cycles=2, L_cycles=1, refinement_steps=2)
    config["training"].update(batch_size=3, max_epochs=2, patience=2, device="cpu")
    config_path = output / "smoke_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    base = nbformat.read(NOTEBOOK, as_version=4)
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
               OMP_NUM_THREADS="2", TRM_CONFIG_PATH=str(config_path), TRM_DATA_PROJECT_ROOT=str(fixture),
               TRM_OUTPUT_ROOT=str(output / "sessions"), TRM_TRAIN_MODEL="true")
    env.pop("TRM_CHECKPOINT_PATH", None)
    def execute(notebook, name, cwd, current_env):
        notebook = deepcopy(notebook)
        NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(cwd)}}).execute(env=current_env)
        nbformat.write(notebook, output / f"{name}.ipynb")
    execute(base, "training_from_root", PROJECT, env)
    checkpoint = next((output / "sessions").glob("*/checkpoints/best.pt"))
    env.update(TRM_TRAIN_MODEL="false", TRM_CHECKPOINT_PATH=str(checkpoint))
    execute(base, "checkpoint_from_notebooks", PROJECT / "notebooks", env)
    runs = sorted((output / "sessions").glob("*/experiment_001/run_001"))
    if len(runs) != 2:
        raise AssertionError("Expected training and checkpoint-only test sessions")
    # Run the actual offline notebook cell with every neural import forbidden.
    offline = nbformat.v4.new_notebook(metadata=base.metadata)
    offline.cells = [nbformat.v4.new_code_cell('''
from pathlib import Path
import builtins
original_import = builtins.__import__
def offline_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'einops'} or name.startswith(('src.trm_model', 'src.trm_adapter', 'src.trm_training')):
        raise RuntimeError('Model import forbidden during offline review: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = offline_import
RUN_DIR = Path(''' + repr(str(runs[0])) + ''')
'''), deepcopy(base.cells[-1])]
    offline.cells.append(nbformat.v4.new_code_cell("builtins.__import__ = original_import"))
    execute(offline, "offline_only", PROJECT / "notebooks", env)
    for run in runs:
        for name in ("predictions", "failures", "evaluation_summary", "coverage_by_severity"):
            if not (run / f"{name}.csv").is_file():
                raise AssertionError(f"Missing export: {name}")
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["status"] != "completed" or manifest["completed_images"] != 4:
            raise AssertionError("Incomplete synthetic notebook run")
    summary = {"status": "passed", "config": config, "runs": [str(p) for p in runs],
               "checks": ["fresh training kernel from project root", "fresh checkpoint-only kernel from notebooks",
                          "model imports forbidden in offline kernel", "four CSV exports per run"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
