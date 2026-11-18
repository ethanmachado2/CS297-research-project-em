"""Grouped fold isolation, out-of-fold coverage and interruption recovery."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image

from src.image_quality_pipeline import build_dataset
from src.recursion_ablation import SCHEMA, VARIANTS
from src.recursion_crossval import (aggregate_oof, cached_data, evaluate_selected,
                                    grouped_splits, main, train_one)
from src.trm_data import create_subject_split, validate_subject_split
from test_vitrm import tiny_config


def fixture(root):
    data = root / "data"
    for severity in range(4):
        directory = data / ("images" if severity == 0 else f"augmentation/{severity}")
        directory.mkdir(parents=True)
        for subject in range(10):
            Image.new("RGB", (32, 32), (severity * 70, subject * 10, 20)).save(directory / f"{subject}.png")
    frame = build_dataset(data)
    reference = root / "reference"
    reference.mkdir()
    split = create_subject_split(frame, 42, project_root=data, counts=[7, 1, 2])
    (reference / "split.json").write_text(json.dumps(split))
    config = tiny_config()
    protocol = dict(checkpoint_schema=SCHEMA, variants=list(VARIANTS), seeds=[42], epochs=1,
        batch_size=8, learning_rate=0.0003, weight_decay=0.0001, gradient_clip=1.0,
        device="cpu", precision="float32", threads=2, dataset_root=str(data),
        model_config={**config["model"], "refinement_steps": 4}, preprocessing=config["preprocessing"])
    (reference / "protocol.json").write_text(json.dumps(protocol))
    return frame, data, reference


class CrossValidationTests(unittest.TestCase):
    def test_grouped_folds_cover_subjects_once_and_preserve_original_first_fold(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame, data, _ = fixture(Path(temporary))
            splits = grouped_splits(frame, data, validation_subjects=1)
            original = create_subject_split(frame, 42, project_root=data, counts=[7, 1, 2])
            self.assertEqual(splits[0]["partitions"], original["partitions"])
            tests = []
            for index, split in enumerate(splits):
                self.assertEqual(split["fold_index"], index)
                validate_subject_split(split, frame, data)
                groups = [set(value["subject_ids"]) for value in split["partitions"].values()]
                self.assertEqual([len(group) for group in groups], [7, 1, 2])
                self.assertEqual(sum(map(len, groups)), len(set.union(*groups)))
                tests.extend(split["partitions"]["test"]["subject_ids"])
            self.assertEqual(len(tests), len(set(tests)))
            self.assertEqual(set(tests), set(frame.subject_id))
            splits[0]["partitions"]["test"]["subject_ids"][0] = splits[0]["partitions"]["train"]["subject_ids"][0]
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_subject_split(splits[0], frame, data)

    def test_resume_keeps_completed_weights_then_pools_every_image_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame, data, reference = fixture(root)
            output = root / "output"
            events, calls = [], []
            def tracked_train(*args, **kwargs):
                calls.append((args[5]["fold_index"], args[0], args[1]))
                if len(calls) == 3:
                    args[4].mkdir()
                    (args[4] / "partial.txt").write_text("interrupted fit")
                    raise KeyboardInterrupt()
                result = train_one(*args, **kwargs)
                events.append("selected")
                return result
            def tracked_data(partition, preprocessing, cache):
                events.append(("pixels", set(partition.subject_id)))
                return cached_data(partition, preprocessing, cache)
            def tracked_evaluation(*args, **kwargs):
                self.assertEqual(events.count("selected"), 15)
                events.append("test_predictions")
                return evaluate_selected(*args, **kwargs)
            with patch("src.recursion_crossval.snapshot_source", return_value={}), \
                 patch("src.recursion_crossval.cached_data", side_effect=tracked_data), \
                 patch("src.recursion_crossval.train_one", side_effect=tracked_train), \
                 patch.object(sys, "argv", ["run", "--reference-run", str(reference), "--output", str(output)]):
                with self.assertRaises(KeyboardInterrupt):
                    main()
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "interrupted")
            retained = {path: path.read_bytes() for path in output.glob("fold_*/*/best.pt")}
            self.assertEqual(len(retained), 2)
            first = json.loads((output / "fold_01/split.json").read_text())
            train_ids = set(first["partitions"]["train"]["subject_ids"])
            val_ids = set(first["partitions"]["validation"]["subject_ids"])
            self.assertEqual(events[:2], [("pixels", train_ids), ("pixels", val_ids)])
            with patch("src.recursion_crossval.cached_data", side_effect=tracked_data), \
                 patch("src.recursion_crossval.train_one", side_effect=tracked_train), \
                 patch("src.recursion_crossval.evaluate_selected", side_effect=tracked_evaluation), \
                 patch.object(sys, "argv", ["run", "--output", str(output), "--resume"]):
                main()
            self.assertEqual(len(calls), 16)  # Fifteen fits plus the interrupted attempt.
            self.assertEqual(retained, {path: path.read_bytes() for path in retained})
            self.assertEqual(len(list((output / "fold_01").glob("interrupted_untied_4_*"))), 1)
            status = json.loads((output / "run.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["completed_training_runs"], 15)
            predictions = pd.read_csv(output / "oof_predictions.csv")
            self.assertEqual(len(predictions), 40 * 7)
            self.assertFalse(predictions.duplicated(["variant", "seed", "step", "image_path"]).any())
            metrics = pd.read_csv(output / "oof_prefix_metrics.csv")
            self.assertEqual(metrics.images.tolist(), [40] * 7)
            self.assertEqual(metrics.subjects.tolist(), [10] * 7)
            transitions = pd.read_csv(output / "oof_transition_metrics.csv")
            self.assertEqual(len(transitions), 6)
            for row in transitions.itertuples():
                self.assertEqual(row.corrections + row.regressions + row.unchanged_correct + row.unchanged_incorrect, 40)
                early = metrics.loc[(metrics.variant == row.variant) & (metrics.step == row.from_step)].iloc[0]
                late = metrics.loc[(metrics.variant == row.variant) & (metrics.step == row.to_step)].iloc[0]
                self.assertEqual(round(40 * (late.severity_accuracy - early.severity_accuracy)), row.net_corrections)
            with patch("src.recursion_crossval.train_one", side_effect=AssertionError("completed fits must not rerun")), \
                 patch.object(sys, "argv", ["run", "--output", str(output), "--resume"]):
                main()
            # Duplicate held-out predictions must fail rather than inflate coverage.
            path = output / "fold_01/prefix_predictions.csv"
            values = pd.read_csv(path)
            pd.concat([values, values.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                aggregate_oof(output, frame, [42], 5)


if __name__ == "__main__":
    unittest.main()
