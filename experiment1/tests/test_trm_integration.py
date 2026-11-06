import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.image_quality_pipeline import run_inference, load_results, validate_prediction, InferenceResult
from src.trm_adapter import TRMImageModel, evaluate_checkpoint, severity_prediction
from src.trm_training import train_experiment
from src.trm_reporting import export_saved_run
from trm_fixtures import make_dataset, tiny_config, PROJECT


class TRMIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name).resolve()
        cls.frame = make_dataset(cls.root)
        cls.path = train_experiment(cls.frame, cls.root, cls.root / "train", tiny_config(), show_progress=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.out = tempfile.TemporaryDirectory()
        self.addCleanup(self.out.cleanup)
        self.output = Path(self.out.name)
        self.adapter = TRMImageModel.from_checkpoint(self.path, device="cpu")

    def test_each_severity_json_mapping(self):
        for severity in range(4):
            logits = torch.full((1, 4), -1.0)
            logits[0, severity] = 1
            with patch.object(self.adapter.model, "forward", return_value=logits):
                raw = self.adapter.generate(self.frame.file_path.iloc[0], "unused prompt")
            prediction = json.loads(raw)
            self.assertEqual(prediction, severity_prediction(severity))
            self.assertEqual(set(prediction), {"noise", "severity", "route"})
            self.assertEqual(validate_prediction(prediction), prediction)

    def test_checkpoint_evaluation_saved_artifacts_and_reload(self):
        run = evaluate_checkpoint(self.path, self.frame, self.root, self.output, device="cpu", show_progress=False)
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["inference_mode"], "supervised")
        self.assertEqual(manifest["total_images"], 4)
        info = manifest["inference_config"]
        for key in ("checkpoint_sha256", "upstream_sha", "preprocessing", "split_digest", "parameter_counts"):
            self.assertIn(key, info)
        self.assertFalse(info["uses_prompt"])
        self.assertEqual(len(load_results(run)), 4)
        self.assertEqual(len(json.loads((run / "labels.json").read_text())), 4)
        self.assertTrue(export_saved_run(run)["complete"])
        for name in ("predictions", "failures", "evaluation_summary", "coverage_by_severity"):
            self.assertTrue((run / f"{name}.csv").is_file())
        with self.assertRaises(FileExistsError):
            evaluate_checkpoint(self.path, self.frame, self.root, self.output, device="cpu", show_progress=False)

    def test_examples_rejected_before_folder_creation(self):
        with self.assertRaisesRegex(ValueError, "does not accept examples"):
            run_inference(self.adapter, [], inference_mode="supervised", examples=[object()], output_dir=self.output / "absent")
        self.assertFalse((self.output / "absent").exists())

    def test_bad_image_nonfinite_and_valid_images_continue(self):
        corrupt = self.output / "corrupt.png"
        corrupt.write_text("not an image")
        paths = [str(corrupt), self.frame.file_path.iloc[0], self.frame.file_path.iloc[1]]
        labels = {path: severity_prediction(0) for path in paths}
        with patch.object(self.adapter.model, "forward", side_effect=[torch.full((1, 4), float("nan")), torch.zeros(1, 4)]):
            results = run_inference(self.adapter, paths, labels=labels, output_dir=self.output,
                                    inference_mode="supervised", show_progress=False)
        self.assertEqual([r.prediction is None for r in results], [True, True, False])
        report = export_saved_run(self.output / "experiment_001/run_001")
        self.assertEqual(report["metrics"]["severity"]["failed"], 2)
        self.assertEqual(report["metrics"]["severity"]["coverage"], 1 / 3)

    def test_interrupted_inference_offline_has_partial_coverage(self):
        paths = self.frame.file_path.iloc[:3].tolist()
        with patch.object(self.adapter, "generate", side_effect=[json.dumps(severity_prediction(0)), KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                run_inference(self.adapter, paths, labels=self.frame.iloc[:3], output_dir=self.output,
                              inference_mode="supervised", show_progress=False)
        report = export_saved_run(self.output / "experiment_001/run_001")
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["attempted"], 1)
        self.assertEqual(report["planned"], 3)
        self.assertEqual(report["metrics"]["severity"]["label_coverage"], 1 / 3)

    def test_hand_calculated_metrics_include_failed_and_unattempted(self):
        from src.evaluation import evaluate
        labels = {str(i): severity_prediction(severity) for i, severity in enumerate([0, 1, 2, 3, 0])}
        results = [InferenceResult(str(i), "", severity_prediction(severity)) for i, severity in enumerate([0, 2, 2, 1])]
        results.append(InferenceResult("4", "bad", None, "error"))
        metrics = evaluate(results, labels)
        values = metrics["severity"]
        for key, expected in (("evaluated", 4), ("failed", 1), ("coverage", 4 / 5), ("accuracy", 0.5),
                              ("mae", 0.75), ("macro_f1", 5 / 12)):
            self.assertAlmostEqual(values[key], expected)
        self.assertEqual(metrics["noise"]["accuracy"], 1)
        self.assertEqual(metrics["route"]["accuracy"], 0.5)
        self.assertEqual(values["correct"] / len(results), 2 / 5)
        labels["5"] = severity_prediction(0)
        self.assertEqual(evaluate(results, labels)["severity"]["label_coverage"], 4 / 6)

    def test_fresh_process_checkpoint_and_model_free_offline_export(self):
        run = evaluate_checkpoint(self.path, self.frame, self.root, self.output, device="cpu", show_progress=False)
        script = '''
import sys
from pathlib import Path
import torch
torch.set_num_threads(2)
from src.trm_adapter import TRMImageModel
model = TRMImageModel.from_checkpoint(sys.argv[1], 'cpu')
assert model.generate(sys.argv[2], 'metadata')
'''
        result = subprocess.run([sys.executable, "-c", script, str(self.path), self.frame.file_path.iloc[0]],
                                cwd=PROJECT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        offline = '''
import sys
from unittest.mock import patch
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','transformers','einops'} or name.startswith(('src.trm_model','src.trm_adapter','src.trm_training')):
        raise RuntimeError('Neural model import forbidden: '+name)
    return original(name, *args, **kwargs)
with patch('builtins.__import__', blocked):
    from src.trm_reporting import export_saved_run
    assert export_saved_run(sys.argv[1])['complete']
'''
        result = subprocess.run([sys.executable, "-c", offline, str(run)], cwd=PROJECT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_modified_split_dataset_rejected_before_test_inference(self):
        frame = self.frame.copy()
        frame.loc[0, "subject_id"] = "different-person"
        with self.assertRaises(ValueError):
            evaluate_checkpoint(self.path, frame, self.root, self.output, device="cpu", show_progress=False)
        self.assertFalse((self.output / "experiment_001").exists())
