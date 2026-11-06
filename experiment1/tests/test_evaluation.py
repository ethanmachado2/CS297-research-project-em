import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation import (
    EvaluationConfig, decision_stability, evaluate, evaluate_saved_run,
    evaluate_stability, run_stability_experiment,
)
from src.image_quality_pipeline import InferenceResult, run_inference


def prediction(severity):
    return {'noise': '0' if severity == 0 else '1', 'severity': str(severity),
            'route': '1' if severity >= 2 else '0'}


def result(path, severity):
    return InferenceResult(path, json.dumps(prediction(severity)), prediction(severity))


class EvaluationTests(unittest.TestCase):
    def test_known_metrics_and_matrix_orientation(self):
        labels = {str(i): prediction(i) for i in range(4)}
        results = [result(str(i), p) for i, p in enumerate([0, 2, 2, 1])]
        metrics = evaluate(results, labels)
        self.assertEqual(metrics['noise']['accuracy'], 1)
        self.assertEqual(metrics['noise']['macro_f1'], 1)
        self.assertEqual(metrics['severity']['confusion_matrix'],
                         [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
        self.assertEqual(metrics['severity']['mae'], .75)
        self.assertAlmostEqual(metrics['severity']['precision'], .375)
        self.assertAlmostEqual(metrics['severity']['recall'], .5)
        self.assertAlmostEqual(metrics['severity']['macro_f1'], 5/12)
        self.assertEqual(metrics['route']['routing_accuracy'], .5)
        self.assertEqual(metrics['route']['precision'], .5)
        self.assertEqual(metrics['route']['recall'], .5)
        self.assertEqual(metrics['route']['macro_f1'], .5)

    def test_failures_missing_labels_empty_and_absent_classes(self):
        labels = {'a': prediction(0), 'b': prediction(0), 'missing': prediction(0)}
        results = [result('a', 0), InferenceResult('b', 'bad', None, 'bad'), result('unknown', 1)]
        metrics = evaluate(results, labels)
        self.assertEqual(metrics['noise']['coverage'], 1/3)
        self.assertEqual(metrics['noise']['label_coverage'], 1/3)
        self.assertEqual(metrics['noise']['failed'], 1)
        self.assertEqual(metrics['noise']['unlabeled'], 1)
        self.assertEqual(metrics['noise']['precision'], 0)
        self.assertEqual(metrics['noise']['macro_f1'], .5)
        self.assertEqual(metrics['severity']['macro_f1'], .25)
        self.assertIsNone(evaluate([], labels)['severity']['mae'])
        with self.assertRaises(ValueError):
            evaluate([result('a', 0), result('a', 1)], labels)
        with self.assertRaises(ValueError):
            evaluate([result('a', 0)], {'a': {**prediction(0), 'severity': '5'}})

    def test_stability_aligns_paths_and_distinguishes_failures(self):
        passes = [
            [result('a', 1), result('b', 0), result('c', 0)],
            [result('c', 0), result('b', 2), result('a', 2)],
            [result('b', 0), InferenceResult('c', 'bad', None, 'invalid'), result('a', 2)],
        ]
        report = decision_stability(passes, image_paths=['a', 'b', 'c', 'missing'])
        noise = report['tasks']['noise']
        route = report['tasks']['route']
        self.assertEqual(noise['stable_images'], 1)
        self.assertEqual(noise['incomplete_images'], 2)
        self.assertEqual(noise['pairwise_agreement'], 5/7)
        self.assertEqual(route['stable_images'], 0)
        self.assertEqual(route['changed_images'], 2)
        self.assertEqual(route['pairwise_agreement'], 3/7)
        self.assertEqual(route['coverage'], .5)
        self.assertEqual(route['stable_fraction'], 0)
        self.assertIsNone(route['per_image'][-1]['pairwise_agreement'])
        with self.assertRaises(ValueError):
            decision_stability(passes[:1])
        with self.assertRaises(ValueError):
            decision_stability([[result('a', 0)] * 2, []])

    def test_three_pass_experiment_saved_reload_and_fresh_calls(self):
        class Changing:
            def __init__(self):
                self.calls = 0
            def generate_batch(self, paths, prompt):
                severity = [0, 2, 0][self.calls % 3]
                self.calls += 1
                return [json.dumps(prediction(severity)) for _ in paths]
        with tempfile.TemporaryDirectory() as folder:
            model = Changing()
            config = EvaluationConfig(output_dir=folder, experiment_number=8)
            report = run_stability_experiment(model, iter(['a', 'b']),
                                             labels={'a': prediction(0), 'b': prediction(2)},
                                             config=config, batch_size=2, show_progress=False)
            self.assertEqual(model.calls, 3)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(len(report['metrics']), 3)
            self.assertEqual(report['stability']['tasks']['route']['changed_images'], 2)
            self.assertEqual(json.loads(Path(report['report_path']).read_text()), {k: v for k, v in report.items() if k != 'report_path'})
            runs = report['run_dirs']
            self.assertEqual(evaluate_saved_run(runs[0]), report['metrics'][0])
            self.assertEqual(evaluate_stability(runs), report['stability'])
            with self.assertRaises(ValueError):
                evaluate_stability([runs[0], runs[0]])
            manifest = Path(runs[1]) / 'manifest.json'
            payload = json.loads(manifest.read_text())
            payload['prompt'] = 'different'
            manifest.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                evaluate_stability(runs)

    def test_interruption_checkpoints_group_and_pass(self):
        calls = []
        def interrupted(path, prompt):
            calls.append(path)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return json.dumps(prediction(0))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(KeyboardInterrupt):
                run_stability_experiment(interrupted, ['a'], labels={'a': prediction(0)},
                                         config=EvaluationConfig(folder), show_progress=False)
            group = Path(folder) / 'experiment_001/stability_001/report.json'
            report = json.loads(group.read_text())
            self.assertEqual(report['status'], 'interrupted')
            self.assertEqual(len(report['metrics']), 1)
            stability = evaluate_stability(report['run_dirs'])
            self.assertEqual(stability['tasks']['route']['incomplete_images'], 1)
            self.assertIsNone(stability['tasks']['route']['pairwise_agreement'])

    def test_few_shot_holdout_preserved_each_pass(self):
        from src.image_quality_pipeline import prepare_few_shot_split
        from test_prompt_modes import dataset, Model
        frame = dataset()
        _, examples = prepare_few_shot_split(frame)
        model = Model()
        with tempfile.TemporaryDirectory() as folder:
            report = run_stability_experiment(model, frame.file_path, labels=frame,
                                             config=EvaluationConfig(folder), examples=examples,
                                             inference_mode='few_shot', batch_size=4, show_progress=False)
            self.assertEqual(len(model.calls), 3)
            self.assertTrue(all(len(call[2]) == 4 for call in model.calls))
            self.assertEqual(report['stability']['total_images'], 4)
            self.assertEqual(report['stability']['tasks']['joint']['stable_fraction'], 1)

    def test_duplicate_ground_truth_rejected_before_generation(self):
        import pandas as pd
        frame = pd.DataFrame([{'file_path': 'a', **prediction(0)}] * 2)
        calls = []
        def model(path, prompt):
            calls.append(path)
            return json.dumps(prediction(0))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                run_inference(model, ['a'], labels=frame, output_dir=folder, show_progress=False)
            with self.assertRaises(ValueError):
                run_stability_experiment(model, ['a'], labels=frame,
                                         config=EvaluationConfig(folder), show_progress=False)
            self.assertEqual(calls, [])
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_empty_and_held_out_only_stability_rejected(self):
        from src.image_quality_pipeline import FewShotExample
        examples = [FewShotExample(f'{i}/reference.jpg', 'reference', prediction(i)) for i in range(4)]
        with tempfile.TemporaryDirectory() as folder:
            for paths, labels, options in [([], {}, {}),
                    ([examples[0].image_path], {examples[0].image_path: prediction(0)}, {'examples': examples})]:
                with self.assertRaises(ValueError):
                    run_stability_experiment(None, paths, labels=labels,
                                             config=EvaluationConfig(folder), **options)
            self.assertEqual(list(Path(folder).iterdir()), [])
        for runs in ([], ['not-a-run']):
            with self.assertRaises(ValueError):
                evaluate_stability(runs)

    def test_stability_normalizes_integer_predictions(self):
        integer = InferenceResult('a', '', {'noise': 0, 'severity': 0, 'route': 0})
        report = decision_stability([[integer], [result('a', 0)]])
        self.assertEqual(report['tasks']['joint']['stable_fraction'], 1)

    def test_saved_adapter_settings_must_match(self):
        class Model:
            inference_config = {'max_new_tokens': 128, 'processor_size': {'longest_edge': 262144}}
            def generate(self, path, prompt):
                return json.dumps(prediction(0))
        with tempfile.TemporaryDirectory() as folder:
            model = Model()
            report = run_stability_experiment(model, ['a'], labels={'a': prediction(0)},
                                             config=EvaluationConfig(folder), show_progress=False)
            run = Path(report['run_dirs'][1])
            manifest_path = run / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest['inference_config'], model.inference_config)
            manifest['inference_config']['max_new_tokens'] = 256
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                evaluate_stability(report['run_dirs'])

    def test_config_validation(self):
        for value in [0, 1, True, 1.5]:
            with self.assertRaises(ValueError):
                EvaluationConfig(stability_passes=value)
        with tempfile.TemporaryDirectory() as folder:
            run_inference(lambda p, q: json.dumps(prediction(0)), ['a'],
                          labels={'a': prediction(0)}, output_dir=folder, show_progress=False)
            self.assertIn('macro_f1', evaluate_saved_run(Path(folder) / 'experiment_001/run_001')['noise'])


if __name__ == '__main__':
    unittest.main()
