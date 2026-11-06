import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from src.image_quality_pipeline import evaluate, load_results, run_inference

CLEAN = '{"noise":"0","severity":"0","route":"0"}'
NOISY = '{"noise":"1","severity":"2","route":"1"}'


class BatchModel:
    def __init__(self):
        self.calls = []

    def generate_batch(self, paths, prompt):
        self.calls.append(list(paths))
        return [CLEAN if path != 'bad' else 'invalid JSON' for path in paths]


class PipelineTests(unittest.TestCase):

    def test_batches_persist_reload_and_evaluate(self):
        with tempfile.TemporaryDirectory() as folder:
            model = BatchModel()
            labels = {p: {'noise': '0', 'severity': '0', 'route': '0'} for p in ['a', 'bad', 'c']}
            results = run_inference(model, iter(labels), batch_size=2, output_dir=folder,
                                    experiment_number=7, labels=labels, show_progress=False)
            run = Path(folder) / 'experiment_007/run_001'
            self.assertEqual(model.calls, [['a', 'bad'], ['c']])
            self.assertEqual(load_results(run), results)
            self.assertEqual(results[1].raw_output, 'invalid JSON')
            self.assertIsNotNone(results[1].error)
            self.assertEqual(json.loads((run / 'metrics.json').read_text()), evaluate(results, labels))
            self.assertEqual(json.loads((run / 'labels.json').read_text()), labels)
            manifest = json.loads((run / 'manifest.json').read_text())
            self.assertEqual((manifest['completed_batches'], manifest['failed_images'], manifest['status']), (2, 1, 'completed'))
            payload = json.loads((run / 'batch_00002.json').read_text())
            self.assertEqual((payload['experiment_number'], payload['run_number'], payload['batch_number']), (7, 1, 2))
            run_inference(model, [], output_dir=folder, experiment_number=7, show_progress=False)
            self.assertTrue((run.parent / 'run_002/manifest.json').exists())
            with self.assertRaises(FileExistsError):
                run_inference(model, ['a'], output_dir=folder, experiment_number=7, run_number=1, show_progress=False)
            self.assertEqual(load_results(run), results)

    def test_legacy_callable_and_single_image_adapter(self):
        def legacy(path, prompt):
            if path == 'bad':
                raise RuntimeError('unreadable')
            return NOISY
        results = run_inference(legacy, ['a', 'bad', 'c'], batch_size=2, output_dir=None, show_progress=False)
        self.assertEqual([r.error for r in results], [None, 'unreadable', None])
        class Single:
            generate = staticmethod(legacy)
        self.assertEqual(run_inference(Single(), ['a'], output_dir=None, show_progress=False)[0], results[0])

    def test_batch_failure_and_wrong_response_count(self):
        class Broken:
            def generate_batch(self, paths, prompt):
                raise RuntimeError('out of memory')
        class Wrong:
            def generate_batch(self, paths, prompt):
                return [CLEAN]
        class String:
            def generate_batch(self, paths, prompt):
                return CLEAN
        for model in [Broken(), Wrong(), String()]:
            with self.subTest(model=model):
                results = run_inference(model, ['a', 'b'], batch_size=2, output_dir=None, show_progress=False)
                self.assertEqual(len(results), 2)
                self.assertTrue(all(r.error for r in results))

    def test_interruption_keeps_completed_batch(self):
        class Interrupted:
            calls = 0
            def generate_batch(self, paths, prompt):
                self.calls += 1
                if self.calls == 2:
                    raise KeyboardInterrupt()
                return [CLEAN]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(KeyboardInterrupt):
                run_inference(Interrupted(), ['a', 'b'], output_dir=folder, show_progress=False)
            run = Path(folder) / 'experiment_001/run_001'
            self.assertEqual([r.image_path for r in load_results(run)], ['a'])
            manifest = json.loads((run / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'interrupted')
            self.assertEqual(manifest['completed_images'], 1)

    def test_empty_run_and_missing_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            results = run_inference(BatchModel(), [], labels={}, output_dir=folder, show_progress=False)
            run = Path(folder) / 'experiment_001/run_001'
            self.assertEqual(results, [])
            self.assertEqual(load_results(run), [])
            self.assertEqual(json.loads((run / 'metrics.json').read_text()), evaluate([], {}))
            with self.assertRaises(FileNotFoundError):
                load_results(run / 'missing')

    def test_validation_before_any_generation(self):
        for key in ['batch_size', 'experiment_number', 'run_number']:
            for value in [0, -1, True, 1.5]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    run_inference(BatchModel(), [], output_dir=None, **{key: value})

    def test_progress_updates_and_closes(self):
        with patch('src.image_quality_pipeline.tqdm') as progress:
            run_inference(BatchModel(), ['a', 'b', 'c'], batch_size=2, output_dir=None)
            bar = progress.return_value.__enter__.return_value
            self.assertEqual([call.args[0] for call in bar.update.call_args_list], [2, 1])
            progress.return_value.__exit__.assert_called_once()
        output = io.StringIO()
        with redirect_stderr(output):
            run_inference(BatchModel(), ['a'], output_dir=None)
        self.assertIn('Inference', output.getvalue())




if __name__ == '__main__':
    unittest.main()
