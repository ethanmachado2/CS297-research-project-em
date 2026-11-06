import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.image_quality_pipeline import (
    FEW_SHOT_GUIDANCE, IMAGE_QUALITY_PROMPT, FewShotExample,
    build_dataset, prepare_few_shot_split, run_inference,
)


def dataset():
    return pd.DataFrame([
        {'file_path': f'{severity}/{subject}.jpg', 'file_name': f'{subject}.jpg',
         'subject_id': subject, 'augmentation': severity, 'severity': severity,
         'noise': '0' if severity == '0' else '1',
         'route': '0' if severity in ('0', '1') else '1'}
        for subject in ['0', '1'] for severity in ['0', '1', '2', '3']
    ])


class Model:
    supports_few_shot = True

    def __init__(self):
        self.calls = []

    def generate_batch(self, paths, prompt, *, examples=()):
        self.calls.append((paths, prompt, examples))
        return ['{"noise":"0","severity":"0","route":"0"}'] * len(paths)


class PromptModeTests(unittest.TestCase):
    def test_subject_split_and_labels(self):
        frame = dataset()
        evaluation, examples = prepare_few_shot_split(frame)
        self.assertEqual(len(frame), 8)
        self.assertEqual(set(evaluation.subject_id), {'1'})
        self.assertEqual([e.prediction['severity'] for e in examples], ['0', '1', '2', '3'])
        self.assertEqual([e.prediction['route'] for e in examples], ['0', '0', '1', '1'])
        self.assertEqual({e.subject_id for e in examples}, {'0'})
        # All images of a subject are excluded, even when more than one image
        # per severity exists under a custom subject grouping.
        extra = frame.iloc[[0]].copy()
        extra['file_path'] = '0/another_capture.jpg'
        evaluation, _ = prepare_few_shot_split(pd.concat([frame, extra]))
        self.assertEqual(len(evaluation), 4)
        evaluation, _ = prepare_few_shot_split(frame.drop(columns='subject_id'))
        self.assertEqual(len(evaluation), 4)

    def test_missing_or_incomplete_subject_rejected(self):
        frame = dataset()
        for invalid in [frame[frame.severity != '3'], frame[frame.subject_id == '0']]:
            with self.assertRaises(ValueError):
                prepare_few_shot_split(invalid)
        with self.assertRaises(ValueError):
            prepare_few_shot_split(frame, 'unknown')

    def test_same_holdout_both_modes_and_saved_metadata(self):
        frame = dataset()
        _, examples = prepare_few_shot_split(frame)
        with tempfile.TemporaryDirectory() as folder:
            for index, mode in enumerate(['zero_shot', 'few_shot'], 1):
                model = Model()
                results = run_inference(model, frame.file_path, labels=frame, examples=examples,
                                        inference_mode=mode, batch_size=3, show_progress=False, output_dir=folder)
                self.assertEqual([r.image_path for r in results], frame[frame.subject_id == '1'].file_path.tolist())
                self.assertEqual(model.calls[0][1], IMAGE_QUALITY_PROMPT + (FEW_SHOT_GUIDANCE if mode == 'few_shot' else ''))
                self.assertEqual(len(model.calls[0][2]), 4 if mode == 'few_shot' else 0)
                run = Path(folder) / f'experiment_001/run_{index:03d}'
                manifest = json.loads((run / 'manifest.json').read_text())
                self.assertEqual(manifest['inference_mode'], mode)
                self.assertEqual(manifest['held_out_subjects'], ['0'])
                self.assertEqual(len(manifest['excluded_image_paths']), 4)
                self.assertEqual(len(manifest['examples']), 4)
                self.assertEqual(len(json.loads((run / 'labels.json').read_text())), 4)
                self.assertEqual(json.loads((run / 'batch_00001.json').read_text())['inference_mode'], mode)

    def test_mode_validation_and_adapter_capability(self):
        _, examples = prepare_few_shot_split(dataset())
        invalid_options = [
            {'inference_mode': 'typo'},
            {'inference_mode': 'few_shot'},
            {'inference_mode': 'few_shot', 'examples': examples[:3]},
            {'examples': [replace(examples[0], subject_id='other'), *examples[1:]]},
            {'examples': [examples[0]] * 4},
        ]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_inference(Model(), [], output_dir=None, **options)
        with self.assertRaises(TypeError):
            run_inference(lambda p, q: '', [], inference_mode='few_shot', examples=examples, output_dir=None)

    def test_custom_subject_ids_and_relative_absolute_aliases(self):
        frame = dataset()
        frame.loc[frame.subject_id == '0', 'subject_id'] = 'reference'
        frame.loc[frame.subject_id == '1', 'subject_id'] = 'evaluation'
        _, examples = prepare_few_shot_split(frame, 'reference')
        paths = frame.file_path.tolist() + [str(Path(examples[0].image_path).resolve())]
        results = run_inference(Model(), paths, labels=frame, examples=examples,
                                inference_mode='few_shot', output_dir=None, show_progress=False)
        self.assertEqual(len(results), 4)

    def test_single_adapter_receives_examples(self):
        class Single:
            supports_few_shot = True
            def generate(self, path, prompt, *, examples):
                self.examples = examples
                return '{"noise":"0","severity":"0","route":"0"}'
        model = Single()
        _, examples = prepare_few_shot_split(dataset())
        results = run_inference(model, ['1/1.jpg'], inference_mode='few_shot', examples=examples,
                                output_dir=None, show_progress=False)
        self.assertEqual(len(model.examples), 4)
        self.assertIsNone(results[0].error)


if __name__ == '__main__':
    unittest.main()
