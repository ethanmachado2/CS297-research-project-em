"""Contracts needed to interpret the controlled moment-encoder comparison."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from src.moment_ablation import (VARIANTS, feature_digest, make_model,
                                 mean_supervised_loss, train_one)

SMALL = dict(hidden_size=16, num_heads=2, layers=1, expansion=2.0,
             memory_tokens=2, token_grid=1, memory_updates=1, refinement_steps=2)
OLD = dict(hidden_size=16, num_heads=2, L_layers=1, expansion=2.0,
           H_cycles=2, L_cycles=2, refinement_steps=4)
torch.set_num_threads(2)


class MomentAblationTests(unittest.TestCase):
    def test_identical_initial_features_shapes_gradients_and_roundtrip(self):
        digests = []
        pixels, labels = torch.rand(3, 1, 3, 16, 16), torch.tensor([0, 1, 3])
        for variant in VARIANTS:
            model = make_model(variant, 42, SMALL, patch_count=1, old_config=OLD)
            digests.append(feature_digest(model))
            logits = model.forward_steps(pixels)
            expected_steps = 4 if variant == "old_trm" else 2 if variant == "vitrm_2" else 1
            self.assertEqual(logits.shape, (3, expected_steps, 4))
            loss = mean_supervised_loss(logits, labels)
            expected = torch.stack([torch.nn.functional.cross_entropy(step, labels)
                                    for step in logits.unbind(dim=1)]).mean()
            torch.testing.assert_close(loss, expected)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            replica = make_model(variant, 43, SMALL, patch_count=1, old_config=OLD)
            replica.load_state_dict(model.state_dict())
            with torch.no_grad():
                torch.testing.assert_close(model(pixels), replica(pixels))
        self.assertEqual(len(set(digests)), 1)

    def test_old_wrapper_uses_unmodified_original_refinement_forward(self):
        model = make_model("old_trm", 42, SMALL, patch_count=1, old_config=OLD)
        pixels = torch.rand(2, 1, 3, 16, 16)
        with torch.no_grad():
            encoded = model.encode(pixels)
            state = model.core.initial_state(2)
            expected = []
            for _ in range(OLD["refinement_steps"]):
                state, logits = model.core.refine_once(encoded, state)
                self.assertFalse(state.answer.requires_grad)
                expected.append(logits)
            torch.testing.assert_close(model.forward_steps(pixels), torch.stack(expected, dim=1))

    def test_training_counts_one_update_per_batch_for_all_variants(self):
        pixels, labels = torch.rand(5, 1, 3, 16, 16), torch.tensor([0, 1, 2, 3, 0])
        data = TensorDataset(pixels, labels)
        protocol = dict(epochs=2, batch_size=3, learning_rate=0.0003, weight_decay=0.0001, gradient_clip=1.0)
        real_factory = make_model
        def small_factory(variant, seed):
            return real_factory(variant, seed, SMALL, patch_count=1, old_config=OLD)
        with tempfile.TemporaryDirectory() as temporary, patch("src.moment_ablation.make_model", small_factory):
            for variant in VARIANTS:
                result = train_one(variant, 42, data, data, Path(temporary) / variant, protocol)
                self.assertEqual(result["optimizer_updates"], 4)
                checkpoint = torch.load(Path(temporary) / variant / "best.pt", weights_only=True)
                self.assertIn(checkpoint["epoch"], [1, 2])
                self.assertEqual(checkpoint["initial_feature_digest"], result["initial_feature_digest"])


if __name__ == "__main__":
    unittest.main()
