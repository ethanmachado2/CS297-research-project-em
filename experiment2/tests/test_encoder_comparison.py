"""Contracts for the focused encoder/recursion comparison."""
import json
from pathlib import Path
import unittest

import torch

from src.encoder_comparison import (VARIANTS, DirectMomentClassifier,
                                    LinearCellEncoder, make_model)

PROJECT = Path(__file__).resolve().parents[1]


def small_config():
    config = json.loads((PROJECT / "configs/vitrm.json").read_text())
    config["preprocessing"].update(patch_size=16, grid_size=1)
    config["model"].update(hidden_size=16, num_heads=2, layers=1,
                           token_grid=2, memory_tokens=2, refinement_steps=2)
    return config


class EncoderComparisonTests(unittest.TestCase):
    def test_linear_encoder_preserves_spatial_token_layout(self):
        encoder = LinearCellEncoder(width=16, grid=2, patch_size=16)
        tokens = encoder(torch.rand(3, 1, 3, 16, 16))
        self.assertEqual(tokens.shape, (3, 4, 16))

    def test_three_variants_return_compatible_logits(self):
        config = small_config()
        pixels = torch.rand(3, 1, 3, 16, 16)
        for variant in VARIANTS:
            model = make_model(variant, config)
            logits = (model.forward_steps(pixels) if variant != "moment_direct"
                      else model(pixels)[:, None])
            self.assertEqual(logits.shape, (3, 2 if variant != "moment_direct" else 1, 4))

    def test_depth_suffix_controls_recursive_unroll(self):
        config = small_config()
        pixels = torch.rand(2, 1, 3, 16, 16)
        for depth in (1, 2, 4):
            model = make_model(f"linear_vitrm_{depth}", config)
            self.assertEqual(model.config["refinement_steps"], depth)
            self.assertEqual(model.forward_steps(pixels).shape, (2, depth, 4))

    def test_direct_classifier_has_no_reasoning_stack(self):
        model = make_model("moment_direct", small_config())
        self.assertIsInstance(model, DirectMomentClassifier)
        self.assertEqual(model.parameter_counts()["reasoning"], 0)


if __name__ == "__main__":
    unittest.main()
