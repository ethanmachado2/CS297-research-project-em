import unittest

import torch

from src.trm_diagnostics import (centroid_probe, compare_gradient_truncation, gradient_trace,
                                 ridge_probe, summarize_features, trace_features)
from src.trm_model import VisionTRM
from trm_fixtures import tiny_config


class TRMDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = VisionTRM(tiny_config()["model"], patch_count=1).eval()
        self.patches = torch.rand(4, 1, 3, 8, 8)
        self.targets = torch.arange(4)

    def test_trace_matches_real_forward_and_cleans_hooks(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        with torch.inference_mode():
            expected = self.model(self.patches)
        features, probabilities = trace_features(self.model, self.patches)
        torch.testing.assert_close(features["outer/2_logits"], expected)
        torch.testing.assert_close(probabilities[-1], expected.softmax(-1))
        self.assertEqual(len([key for key in features if key.endswith("_readout")]), 8)
        self.assertEqual(features["encoder/projected_patch_mean"].shape, (4, 16))
        self.assertFalse(self.model.reasoner._forward_hooks)
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.encoder.conv))
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_gradients_measure_real_loss_without_updating_parameters(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        result = gradient_trace(self.model, self.patches, self.targets)
        self.assertEqual(len(result), 2)
        self.assertGreater(result[0]["encoded_patch_gradient_norm"], 0)
        for step in result:
            for group in step["groups"].values():
                self.assertEqual(group["missing_gradients"], 0)
                self.assertGreater(group["gradient_norm"], 0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_fixed_probe_recovers_signal_and_does_not_fit_validation_labels(self):
        train = torch.eye(4).repeat(3, 1)
        targets = torch.arange(4).repeat(3)
        result = ridge_probe(train, targets, torch.eye(4), torch.arange(4))
        shuffled_labels = ridge_probe(train, targets, torch.eye(4), torch.tensor([1, 2, 3, 0]))
        self.assertEqual(result["train"]["severity_accuracy"], 1)
        self.assertEqual(result["validation"]["severity_accuracy"], 1)
        self.assertEqual(shuffled_labels["validation"]["severity_accuracy"], 0)
        self.assertEqual(result["train"], shuffled_labels["train"])
        constant = ridge_probe(torch.ones(12, 3), targets, torch.ones(4, 3), torch.arange(4))
        self.assertEqual(constant["validation"]["severity_accuracy"], .25)

    def test_centroid_probe_handles_ordered_scalar_noise(self):
        x = torch.tensor([0., 1., 2., 3.])[:, None].repeat(3, 1)
        targets = torch.arange(4).repeat(3)
        val = torch.tensor([.01, 1.01, 2.01, 3.01])[:, None]
        result = centroid_probe(x, targets, val, torch.arange(4))
        self.assertEqual(result["validation"]["severity_accuracy"], 1)

    def test_moment_control_starts_with_same_encoder_function(self):
        from scripts.diagnose_trm_controls import EncoderClassifier
        mean = EncoderClassifier(self.model, moments=False)
        moments = EncoderClassifier(self.model, moments=True)
        # Three stride-2 convolutions leave one spatial location on the 8px
        # fixture, whose standard deviation is correctly zero. Use 32px here.
        patches = torch.rand(4, 1, 3, 32, 32)
        torch.testing.assert_close(mean(patches), moments(patches))
        torch.nn.functional.cross_entropy(moments(patches), self.targets).backward()
        self.assertGreater(moments.encoder.projection.weight.grad[:, 64:].norm().item(), 0)

    def test_variation_distinguishes_constant_and_class_dependent_features(self):
        constant = summarize_features(torch.ones(4, 8), self.targets)
        signal = summarize_features(torch.eye(4), self.targets)
        self.assertEqual(constant["image_std_rms"], 0)
        self.assertEqual(constant["between_class_variance_fraction"], 0)
        self.assertAlmostEqual(signal["between_class_variance_fraction"], 1)

    def test_truncation_comparison_keeps_forward_identical(self):
        result = compare_gradient_truncation(self.model, self.patches, self.targets)
        self.assertEqual(result["default_truncated"]["loss"], result["all_inner_cycles"]["loss"])
        self.assertGreater(result["all_inner_cycles"]["encoder_gradient_norm"], 0)
