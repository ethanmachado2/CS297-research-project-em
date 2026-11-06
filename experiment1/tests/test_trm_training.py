import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from src.trm_data import FaceSeverityDataset, partition_frame, load_image_patches
from src.trm_training import train_experiment, load_checkpoint, select_device, validation_metrics
from trm_fixtures import make_dataset, tiny_config, PROJECT


class TRMTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.frame = make_dataset(self.root)
        self.config = tiny_config()
        self.session = self.root / "run"

    def train(self, **kwargs):
        return train_experiment(self.frame, self.root, self.session, self.config, show_progress=False, **kwargs)

    def test_real_autograd_training_remainder_and_cpu_checkpoint(self):
        path = self.train()
        manifest = json.loads((self.session / "training_manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["optimizer_updates"], 12)  # ceil(8/3) * 2 refinements * 2 epochs
        history = pd.read_csv(self.session / "training_history.csv")
        self.assertEqual(history.training_images.tolist(), [8, 8])
        self.assertEqual(history.optimizer_updates.tolist(), [6, 12])
        self.assertTrue(torch.isfinite(torch.tensor(history.training_step_loss.tolist())).all())
        model, checkpoint = load_checkpoint(path)
        self.assertTrue(all(value.device.type == "cpu" for value in checkpoint["model_state"].values()))
        self.assertEqual(checkpoint["training_status"], "completed")
        patches = load_image_patches(self.frame.file_path.iloc[0], self.config["preprocessing"]).unsqueeze(0)
        other, _ = load_checkpoint(path)
        with torch.inference_mode():
            torch.testing.assert_close(model(patches), other(patches))
        with self.assertRaises(FileExistsError):
            self.train()

    def test_selection_only_accesses_train_and_validation_pixels(self):
        accessed, val_modes = [], []
        original = FaceSeverityDataset.__getitem__
        def tracked(dataset, index):
            accessed.append(dataset.frame.iloc[index].file_path)
            return original(dataset, index)
        def validate(model, loader, device):
            before = [p.detach().clone() for p in model.parameters()]
            result = validation_metrics(model, loader, device)
            val_modes.append(model.training)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))
            return result
        with patch.object(FaceSeverityDataset, "__getitem__", tracked), patch("src.trm_training.validation_metrics", side_effect=validate):
            self.train()
        split = json.loads((self.session / "split.json").read_text())
        test_paths = set(partition_frame(split, "test", self.root).file_path)
        self.assertFalse(set(accessed) & test_paths)
        self.assertEqual(set(accessed), set(self.frame.file_path) - test_paths)
        self.assertEqual(val_modes, [False, False])

    def test_validation_earliest_tie_and_patience(self):
        self.config["training"].update(max_epochs=8, patience=2)
        values = [{"loss": loss, "accuracy": 0.25, "macro_f1": 0.1} for loss in (0.8, 0.5, 0.5, 0.6)]
        with patch("src.trm_training.validation_metrics", side_effect=values):
            path = self.train()
        _, checkpoint = load_checkpoint(path)
        self.assertEqual(checkpoint["epoch"], 2)
        manifest = json.loads((self.session / "training_manifest.json").read_text())
        self.assertTrue(manifest["stopped_early"])
        self.assertEqual(manifest["completed_epochs"], 4)

    def test_validation_improvement_tolerance(self):
        self.config["training"].update(max_epochs=3, patience=3)
        values = [{"loss": loss, "accuracy": 0, "macro_f1": 0} for loss in (0.5, 0.4999995, 0.499998)]
        with patch("src.trm_training.validation_metrics", side_effect=values):
            path = self.train()
        _, checkpoint = load_checkpoint(path)
        self.assertEqual(checkpoint["epoch"], 3)

    def test_interruption_retains_history_and_best(self):
        with patch("src.trm_training.validation_metrics", side_effect=[{"loss": 0.5, "accuracy": 0, "macro_f1": 0}, KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                self.train()
        manifest = json.loads((self.session / "training_manifest.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual(len(pd.read_csv(self.session / "training_history.csv")), 1)
        _, checkpoint = load_checkpoint(self.session / "checkpoints/best.pt")
        self.assertEqual(checkpoint["training_status"], "interrupted")

    def test_nonfinite_loss_persists_failed_status(self):
        with patch("src.trm_training.nn.functional.cross_entropy", return_value=torch.tensor(float("nan"))):
            with self.assertRaises(FloatingPointError):
                self.train()
        self.assertEqual(json.loads((self.session / "training_manifest.json").read_text())["status"], "failed")
        self.assertFalse((self.session / "checkpoints/best.pt").exists())

    def test_nonfinite_gradient_persists_failed_status(self):
        from src.trm_model import VisionTRM
        original = VisionTRM.encode
        def bad_gradient(model, patches):
            result = original(model, patches)
            result.register_hook(lambda grad: grad * float("nan"))
            return result
        with patch.object(VisionTRM, "encode", bad_gradient):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                self.train()
        self.assertEqual(json.loads((self.session / "training_manifest.json").read_text())["status"], "failed")

    def test_unavailable_device_fails_without_fallback(self):
        self.config["training"]["device"] = "unavailable-device"
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.train()
        self.assertEqual(json.loads((self.session / "training_manifest.json").read_text())["status"], "failed")

    def test_checkpoint_compatibility_errors(self):
        path = self.train()
        _, checkpoint = load_checkpoint(path)
        for key, value in (("schema_version", 99), ("split_digest", "bad"), ("upstream_sha", "wrong")):
            invalid = dict(checkpoint, **{key: value})
            bad = self.root / "invalid.pt"
            torch.save(invalid, bad)
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_checkpoint(bad)
        invalid = copy.deepcopy(checkpoint)
        invalid["config"]["model"]["H_cycles"] = 0
        torch.save(invalid, bad)
        with self.assertRaises(ValueError):
            load_checkpoint(bad)

    @unittest.skipUnless(os.environ.get("TRM_OVERFIT_DIAGNOSTIC") == "1", "Opt-in synthetic overfit diagnostic")
    def test_tiny_four_class_learnability(self):
        from src.trm_model import VisionTRM
        torch.manual_seed(42)
        config = tiny_config()["model"]
        model = VisionTRM(config, patch_count=1)
        pixels = torch.stack([torch.full((1, 3, 8, 8), value / 3) for value in range(4)])
        targets = torch.arange(4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        losses = []
        for _ in range(150):
            state = model.initial_state(4)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                state, logits = model.refine_once(model.encode(pixels), state)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            losses.append(loss.item())
        with torch.inference_mode():
            accuracy = (model(pixels).argmax(-1) == targets).float().mean().item()
        print(f"Synthetic overfit: initial_loss={losses[0]:.5f}, final_loss={losses[-1]:.5f}, training_accuracy={accuracy}")
        self.assertLess(losses[-1], losses[0] * 0.3)
        self.assertEqual(accuracy, 1.0)
