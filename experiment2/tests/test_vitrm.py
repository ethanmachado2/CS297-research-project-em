"""ViTRM recurrence, noise features, learning, and checkpoint evaluation contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F

from src.image_quality_pipeline import build_dataset
from src.trm_adapter import evaluate_checkpoint
from src.trm_data import FaceSeverityDataset, partition_frame
from src.trm_reporting import export_saved_run
from src.trm_training import build_model, load_checkpoint, train_experiment, validate_config
from src.vitrm_model import IMPLEMENTATION, MomentEncoder, VisionViTRM

PROJECT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def tiny_config(device="cpu", precision="float32"):
    config = json.loads((PROJECT / "configs/vitrm.json").read_text())
    config["split"]["counts"] = [2, 1, 1]
    config["preprocessing"].update(patch_size=16, grid_size=1)
    config["model"].update(hidden_size=16, num_heads=2, layers=1,
                           token_grid=1, memory_tokens=2, refinement_steps=2)
    config["training"].update(device=device, precision=precision, batch_size=3, max_epochs=2, patience=2)
    return config


def make_images(root):
    for severity in range(4):
        folder = root / ("images" if severity == 0 else f"augmentation/{severity}")
        folder.mkdir(parents=True)
        for subject in range(4):
            Image.new("RGB", (32, 32), color=(severity * 70, subject * 10, 20)).save(folder / f"{subject}.png")
    return build_dataset(root)


class ViTRMTests(unittest.TestCase):
    def test_shared_updates_keep_image_tokens_fixed_and_gradients_span_unroll(self):
        torch.manual_seed(42)
        config = tiny_config()["model"]
        model = VisionViTRM(config, patch_count=1)
        images = torch.rand(3, 1, 3, 16, 16)
        inputs = []
        hook = model.reasoner.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
        steps = model.forward_steps(images)
        hook.remove()
        self.assertEqual(steps.shape, (3, 2, 4))
        self.assertEqual([value.shape[1] for value in inputs], [4, 3, 4, 3])
        torch.testing.assert_close(inputs[0][:, :1], inputs[2][:, :1])
        inputs[1].retain_grad()
        F.cross_entropy(steps[:, -1], torch.tensor([0, 1, 3])).backward()
        self.assertGreater(inputs[1].grad.abs().sum().item(), 0)
        for component in (model.encoder, model.reasoner, model.classifier):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters()))
        self.assertGreater(model.memory_init.grad.abs().sum().item(), 0)
        self.assertGreater(model.answer_init.grad.abs().sum().item(), 0)
        with torch.no_grad():
            torch.testing.assert_close(model(images), model.forward_steps(images)[:, -1])
            torch.testing.assert_close(model(images[:1]), model(images)[:1], atol=1e-6, rtol=1e-5)
            self.assertGreater((model(images[:1]) - model(images[1:2])).abs().max().item(), 1e-6)

    def test_moments_distinguish_equal_mean_different_noise_amplitudes(self):
        encoder = MomentEncoder(16, 1)
        checker = (torch.arange(16)[:, None] + torch.arange(16)[None, :]) % 2
        pixels = torch.stack([torch.full((3, 16, 16), 0.5), checker.expand(3, -1, -1).float()])
        moments = encoder.moments(pixels)
        torch.testing.assert_close(moments[0, :3], moments[1, :3])
        self.assertGreater((moments[1, 3:] - moments[0, 3:]).min().item(), 0.49)
        config = json.loads((PROJECT / "configs/vitrm.json").read_text())
        model = build_model(config)
        with torch.no_grad():
            self.assertEqual(model.encode(torch.rand(1, 9, 3, 128, 128)).shape, (1, 144, 128))

    def test_noise_amplitude_subset_is_learnable(self):
        torch.manual_seed(7)
        model = build_model(tiny_config())
        # Balanced zero-mean texture: classes differ in noise amplitude, not mean.
        checker = ((torch.arange(16)[:, None] + torch.arange(16)[None, :]) % 2).float() * 2 - 1
        labels = torch.arange(4).repeat(2)
        images = torch.stack([0.5 + checker.expand(1, 3, -1, -1) * (0.02 + 0.12 * int(label))
                              for label in labels])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0)
        initial = F.cross_entropy(model(images), labels).item()
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_steps(images)
            loss = F.cross_entropy(logits.flatten(0, 1), labels[:, None].expand(-1, 2).reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        with torch.no_grad():
            final = model(images)
            self.assertLess(F.cross_entropy(final, labels).item(), initial * 0.2)
            self.assertTrue(torch.equal(final.argmax(-1), labels))

    def test_config_rejects_unsupported_architecture_and_impossible_grid(self):
        config = tiny_config()
        validate_config(config)
        invalid = copy.deepcopy(config)
        invalid["architecture"] = "typo"
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_config(invalid)
        invalid = copy.deepcopy(config)
        invalid["model"]["token_grid"] = 3
        with self.assertRaisesRegex(ValueError, "feature map"):
            validate_config(invalid)

    def exercise_training_and_evaluation(self, device, precision):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = root / "data"
            frame = make_images(data)
            config = tiny_config(device, precision)
            accessed = []
            original = FaceSeverityDataset.__getitem__
            def tracked(dataset, index):
                accessed.append(dataset.frame.iloc[index].file_path)
                return original(dataset, index)
            with patch.object(FaceSeverityDataset, "__getitem__", tracked):
                path = train_experiment(frame, data, root / "training", config, show_progress=False)
            model, checkpoint = load_checkpoint(path, device)
            self.assertEqual(checkpoint["implementation"], IMPLEMENTATION)
            self.assertIsNone(checkpoint["upstream_sha"])
            held_out = partition_frame(checkpoint["split"], "test", data)
            self.assertFalse(set(accessed) & set(held_out.file_path))
            history = pd.read_csv(root / "training/training_history.csv")
            if precision == "float32":
                self.assertEqual(history.optimizer_updates.tolist(), [3, 6])
            else:
                self.assertEqual(int(history.optimizer_updates.iloc[-1] + history.skipped_updates.iloc[-1]), 6)
                self.assertGreater(history.optimizer_updates.iloc[-1], 0)
            self.assertTrue((history.encoder_gradient_norm > 0).all())
            self.assertIn("validation_step_2_probability_std", history)
            model2, _ = load_checkpoint(path, device)
            pixels = torch.rand(2, 1, 3, 16, 16, device=device)
            with torch.inference_mode():
                torch.testing.assert_close(model(pixels), model2(pixels))
            # CPU evaluation remains available for a checkpoint trained with AMP.
            run = evaluate_checkpoint(path, frame, data, root / "evaluation", device="cpu", show_progress=False)
            report = export_saved_run(run)
            self.assertEqual(report["planned"], 4)
            self.assertEqual(len(report["failures"]), 0)
            steps = pd.read_csv(run / "refinement_predictions.csv")
            summary = pd.read_csv(run / "refinement_summary.csv")
            self.assertEqual(len(steps), 8)
            self.assertEqual(summary.valid.tolist(), [4, 4])
            self.assertEqual(summary.severity_accuracy_valid.iloc[-1], report["metrics"]["severity"]["accuracy"])
            from src.trm_data import load_image_patches
            failed_image = held_out.file_path.iloc[0]
            def failing_image(path, preprocessing):
                if path == failed_image:
                    raise OSError("fixture decode failure")
                return load_image_patches(path, preprocessing)
            with patch("src.trm_adapter.load_image_patches", side_effect=failing_image):
                failed_run = evaluate_checkpoint(path, frame, data, root / "partial", device="cpu", show_progress=False)
            failed_report = export_saved_run(failed_run)
            failed_summary = pd.read_csv(failed_run / "refinement_summary.csv")
            self.assertEqual(len(failed_report["failures"]), 1)
            self.assertEqual(failed_summary.valid.tolist(), [3, 3])
            self.assertEqual(failed_summary.coverage.tolist(), [0.75, 0.75])
            self.assertEqual(failed_summary.severity_correct_planned.iloc[-1],
                             failed_report["summary"].loc["severity", "Correct / planned"])
            changed = copy.deepcopy(frame)
            Image.new("RGB", (32, 32), "white").save(changed.file_path.iloc[0])
            with self.assertRaisesRegex(ValueError, "changed"):
                evaluate_checkpoint(path, changed, data, root / "invalid", device="cpu", show_progress=False)

    def test_cpu_training_checkpoint_and_held_out_reports(self):
        self.exercise_training_and_evaluation("cpu", "float32")

    def test_cli_validation_only_then_checkpoint_only_evaluation(self):
        from src.run import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            frame = make_images(root / "data")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(tiny_config()))
            common = ["run", "vitrm_supervised", "--device", "cpu"]
            with patch("src.dataset.ensure_dataset", return_value=(root / "data", {"images": len(frame)}, {})), \
                 patch("src.run.snapshot_source", return_value={}):
                with patch.object(sys, "argv", common + ["--config", str(config_path), "--validation-only",
                                                        "--output", str(root / "training")]):
                    main()
                status = json.loads((root / "training/run.json").read_text())
                self.assertFalse(status["test_evaluated"])
                self.assertNotIn("results_directory", status)
                with patch("src.trm_training.train_experiment", side_effect=AssertionError("must not retrain")), \
                     patch.object(sys, "argv", common + ["--checkpoint", status["checkpoint"],
                                                        "--output", str(root / "evaluation")]):
                    main()
                evaluated = json.loads((root / "evaluation/run.json").read_text())
                self.assertTrue(evaluated["evaluation_only"])
                self.assertEqual(evaluated["status"], "completed")
                self.assertEqual(evaluated["planned"], 4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime required")
    def test_cuda_amp_training_checkpoint_and_held_out_reports(self):
        self.exercise_training_and_evaluation("cuda", "float16")


if __name__ == "__main__":
    unittest.main()
