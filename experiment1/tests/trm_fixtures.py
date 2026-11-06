"""Small real pixel/model fixtures shared by the TRM tests."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.image_quality_pipeline import build_dataset

PROJECT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def tiny_config():
    config = json.loads((PROJECT / "configs/trm_experiment1.json").read_text())
    config["split"]["counts"] = [2, 1, 1]
    config["preprocessing"].update(patch_size=8, grid_size=1)
    config["model"].update(hidden_size=16, num_heads=2, L_layers=2, H_cycles=2, L_cycles=1, refinement_steps=2)
    config["training"].update(batch_size=3, max_epochs=2, patience=2, device="cpu")
    return copy.deepcopy(config)


def make_dataset(root, people=4, size=16):
    root = Path(root)
    rng = np.random.default_rng(123)
    for severity in range(4):
        folder = root / "data/files" / ("images" if severity == 0 else f"augmentation/{severity}")
        folder.mkdir(parents=True, exist_ok=True)
        for person in range(people):
            pixels = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{person}.png")
    return build_dataset(root / "data/files")
