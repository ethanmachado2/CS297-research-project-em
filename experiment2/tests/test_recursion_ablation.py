"""Paired depth metrics, truly untied controls, and held-out run sequencing."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from src.moment_ablation import feature_digest, mean_supervised_loss
from src.recursion_ablation import (VARIANTS, collect_logits, main, make_depth_model,
                                    prefix_reports)
from src.trm_data import create_subject_split, partition_frame
from test_vitrm import make_images, tiny_config

torch.set_num_threads(2)


class RecursionAblationTests(unittest.TestCase):
    def test_same_initial_function_untied_weights_and_prefixes(self):
        config = tiny_config()["model"]
        pixels = torch.rand(3, 1, 3, 16, 16)
        models = {name: make_depth_model(name, 42, config, patch_count=1) for name in VARIANTS}
        self.assertEqual(len({feature_digest(model) for model in models.values()}), 1)
        shared, untied = models["shared_4"], models["untied_4"]
        stacks = [stack for stage in untied.reasoner for stack in stage]
        self.assertEqual(len(stacks), 8)
        pointers = [next(stack.parameters()).data_ptr() for stack in stacks]
        self.assertEqual(len(set(pointers)), 8)
        for stack in stacks:
            for expected, actual in zip(shared.reasoner.parameters(), stack.parameters()):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        shared_count = sum(p.numel() for p in shared.reasoner.parameters())
        self.assertEqual(sum(p.numel() for p in untied.reasoner.parameters()), 8 * shared_count)
        with torch.no_grad():
            tokens = shared.encode(pixels)
            before = tokens.clone()
            full = shared.forward_tokens(tokens)
            torch.testing.assert_close(full, untied.forward_tokens(tokens), rtol=0, atol=0)
            torch.testing.assert_close(full[:, :1], models["shared_1"].forward_tokens(tokens), rtol=0, atol=0)
            for depth in (1, 2, 4):
                torch.testing.assert_close(full[:, :depth], shared.forward_tokens(tokens, steps=depth), rtol=0, atol=0)
            torch.testing.assert_close(tokens, before, rtol=0, atol=0)
            for depth in (0, 5, 1.5, True):
                with self.assertRaisesRegex(ValueError, "training depth"):
                    shared.forward_tokens(tokens, steps=depth)
        # Every independent stage learns, including from supervision at late depths.
        for model in (shared, untied):
            mean_supervised_loss(model.forward_steps(pixels), torch.tensor([0, 1, 3])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            early = model.reasoner if model is shared else model.reasoner[0][0]
            self.assertGreater(sum(p.grad.abs().sum().item() for p in early.parameters()), 0)

    def test_corrections_regressions_routes_and_probability_metrics(self):
        labels = torch.arange(4)
        logits = torch.zeros(4, 4, 4)
        predictions = [[0, 0, 2, 3], [1, 1, 0, 3], [0, 0, 0, 0], [0, 1, 2, 2]]
        for index, values in enumerate(predictions):
            logits[torch.arange(4), index, torch.tensor(values)] = 3 + index * 0.2
        frame = pd.DataFrame({"file_path": [f"image{i}" for i in range(4)],
                              "subject_id": ["a", "b", "c", "d"], "severity": list("0123")})
        reports = prefix_reports(logits, labels, frame)
        summaries = reports["prefix_metrics"].set_index("step")
        self.assertEqual(summaries.severity_accuracy.tolist(), [0.75, 0.5, 0.75])
        self.assertEqual(summaries.routing_errors.tolist(), [0, 1, 0])
        self.assertEqual(summaries.route_false_negatives.tolist(), [0, 1, 0])
        self.assertEqual(summaries.route_false_positives.tolist(), [0, 0, 0])
        self.assertEqual(summaries.severity_mae.tolist(), [0.25, 0.75, 0.25])
        for step in (1, 2, 4):
            self.assertAlmostEqual(summaries.loc[step, "cross_entropy"], F.cross_entropy(logits[:, step - 1], labels).item())
        pairs = reports["transition_metrics"].set_index(["from_step", "to_step"])
        self.assertEqual(pairs.loc[(1, 2), "corrections"], 1)
        self.assertEqual(pairs.loc[(1, 2), "regressions"], 2)
        self.assertEqual(pairs.loc[(1, 4), "corrections"], 1)
        self.assertEqual(pairs.loc[(1, 4), "regressions"], 1)
        self.assertNotEqual(pairs.loc[(1, 4), "cross_entropy_delta"], 0)
        self.assertEqual(pairs.loc[(2, 4), "routing_corrections"], 1)
        for (early, late), row in pairs.iterrows():
            self.assertEqual(row.net_corrections,
                round(4 * (summaries.loc[late, "severity_accuracy"] - summaries.loc[early, "severity_accuracy"])))
            self.assertEqual(row.corrections + row.regressions + row.unchanged_correct + row.unchanged_incorrect, 4)
        images = reports["image_transitions"]
        self.assertEqual(len(images), 12)
        self.assertEqual(images.loc[(images.from_step == 1) & (images.to_step == 4)].corrected.sum(), 1)
        with self.assertRaises(ValueError):
            prefix_reports(logits, labels, frame, (1, 5))
        with self.assertRaises(ValueError):
            prefix_reports(logits, labels.flip(0), frame)

    def test_collect_encodes_once_per_batch(self):
        model = make_depth_model("shared_4", 42, tiny_config()["model"], patch_count=1)
        data = TensorDataset(torch.rand(5, 1, 3, 16, 16), torch.arange(5) % 4)
        with patch.object(model, "encode", wraps=model.encode) as encode:
            logits = collect_logits(model, data, batch_size=3)
        self.assertEqual(logits.shape, (5, 4, 4))
        self.assertEqual(encode.call_count, 2)

    def test_validation_only_then_same_checkpoints_without_retraining(self):
        from src.recursion_ablation import preload, train_one
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = make_images(root / "data")
            split = create_subject_split(frame, 42, project_root=root / "data", counts=[2, 1, 1])
            reference = root / "reference.json"
            reference.write_text(json.dumps(split))
            (root / "configs").mkdir()
            (root / "configs/vitrm.json").write_text(json.dumps(tiny_config()))
            output = root / "output"
            events = []
            def tracked_preload(partition, preprocessing):
                events.append(("load", set(partition.file_path)))
                return preload(partition, preprocessing)
            def tracked_train(*args, **kwargs):
                result = train_one(*args, **kwargs)
                events.append(("selected", result["variant"]))
                return result
            arguments = ["run", "--dataset-root", str(root / "data"), "--reference-split", str(reference),
                         "--output", str(output), "--epochs", "2", "--seeds", "42", "--validation-only"]
            with patch("src.recursion_ablation.PROJECT", root), \
                 patch("src.recursion_ablation.snapshot_source", return_value={}), \
                 patch("src.recursion_ablation.preload", side_effect=tracked_preload), \
                 patch("src.recursion_ablation.train_one", side_effect=tracked_train), \
                 patch.object(sys, "argv", arguments):
                main()
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "ready_for_evaluation")
            test_paths = set(partition_frame(split, "test", root / "data").file_path)
            self.assertTrue(all(not value & test_paths for kind, value in events if kind == "load"))
            self.assertEqual([kind for kind, _ in events].count("selected"), 3)
            before = {path: path.read_bytes() for path in output.glob("*/best.pt")}
            self.assertEqual(len(before), 3)
            with patch("src.recursion_ablation.preload", side_effect=tracked_preload), \
                 patch("src.recursion_ablation.train_one", side_effect=AssertionError("must not retrain")), \
                 patch.object(sys, "argv", ["run", "--output", str(output), "--evaluate-only"]):
                main()
            self.assertEqual(events[-1], ("load", test_paths))
            self.assertEqual(before, {path: path.read_bytes() for path in before})
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "completed")
            metrics = pd.read_csv(output / "prefix_metrics.csv")
            self.assertEqual(metrics.groupby("variant").step.apply(list).to_dict(),
                             {"shared_1": [1], "shared_4": [1, 2, 4], "untied_4": [1, 2, 4]})
            transitions = pd.read_csv(output / "image_transitions.csv")
            self.assertEqual(len(transitions), 24)  # Two deep models x three pairs x four images.
            checkpoints = json.loads((output / "selected_checkpoints.json").read_text())
            self.assertTrue(all(row["optimizer_updates"] == 2 for row in checkpoints))
            self.assertTrue(all((group.checkpoint_sha256.nunique() == 1)
                                for _, group in metrics.groupby("variant")))


if __name__ == "__main__":
    unittest.main()
