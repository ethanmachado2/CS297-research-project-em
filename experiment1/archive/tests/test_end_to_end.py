"""Fast pipeline integration using real image files and a deterministic model fixture.

The opt-in scripts/validate_end_to_end.py separately exercises cached Qwen weights.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from PIL import Image

from src.evaluation import EvaluationConfig, evaluate_saved_run, evaluate_stability, run_stability_experiment
from src.image_quality_pipeline import build_dataset, load_results, prepare_few_shot_split
from src.transformers_adapter import TransformersImageModel


def labels(severity):
    return {'noise': str(int(severity > 0)), 'severity': str(severity), 'route': str(int(severity > 1))}


class Inputs(dict):
    def to(self, device):
        return self


class Processor:
    tokenizer = SimpleNamespace(padding_side='right')
    image_processor = SimpleNamespace(size={'longest_edge': 262144, 'shortest_edge': 65536})

    def __init__(self):
        self.turn_counts = []

    def apply_chat_template(self, conversations, **options):
        self.turn_counts.extend(len(conversation) for conversation in conversations)
        assert options['processor_kwargs']['padding']
        assert self.tokenizer.padding_side == 'left'
        values = [conversation[-1]['content'][0]['image'].getpixel((0, 0))[0] // 60
                  for conversation in conversations]
        return Inputs(input_ids=torch.tensor([[9, value] for value in values]))

    def batch_decode(self, tokens, **options):
        return [json.dumps(labels(int(row[0]))) for row in tokens]


class Model:
    device = 'cpu'
    dtype = torch.float32
    config = SimpleNamespace(_commit_hash='fixture', _name_or_path='fixture')

    def eval(self):
        return self

    def generate(self, *, input_ids, **options):
        assert not torch.is_grad_enabled()
        assert options['do_sample'] is False
        return torch.cat((input_ids, input_ids[:, -1:]), dim=1)


class EndToEndTests(unittest.TestCase):
    def test_images_to_saved_metrics_and_cli_in_both_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for severity in range(4):
                directory = root / 'data' / ('images' if severity == 0 else f'augmentation/{severity}')
                directory.mkdir(parents=True)
                for subject in range(3):
                    Image.new('RGB', (16 + subject, 16), (severity * 60, 0, 0)).save(directory / f'{subject}.png')
            frame = build_dataset(root / 'data')
            evaluation, examples = prepare_few_shot_split(frame)
            self.assertEqual(len(frame), 12)
            self.assertEqual(len(evaluation), 8)
            for number, mode in enumerate(('zero_shot', 'few_shot'), 1):
                processor = Processor()
                report = run_stability_experiment(
                    TransformersImageModel(Model(), processor), frame.file_path, labels=frame,
                    config=EvaluationConfig(root / 'outputs', number),
                    batch_size=3, inference_mode=mode, examples=examples, show_progress=False,
                )
                self.assertEqual(len(processor.turn_counts), 24)
                self.assertEqual(set(processor.turn_counts), {9 if mode == 'few_shot' else 1})
                for run_dir, metrics in zip(report['run_dirs'], report['metrics']):
                    run = Path(run_dir)
                    self.assertEqual(len(list(run.glob('batch_*.json'))), 3)
                    self.assertEqual({r.image_path for r in load_results(run)}, set(evaluation.file_path))
                    self.assertEqual(evaluate_saved_run(run, save=False), metrics)
                    for task in ('noise', 'severity', 'route'):
                        self.assertEqual(metrics[task]['accuracy'], 1)
                        self.assertEqual(metrics[task]['macro_f1'], 1)
                        self.assertEqual(metrics[task]['label_coverage'], 1)
                    self.assertEqual(metrics['severity']['mae'], 0)
                cli = subprocess.run([sys.executable, '-m', 'src.evaluation', *report['run_dirs']],
                                     capture_output=True, text=True, check=True)
                self.assertEqual(list(json.loads(cli.stdout).values()), report['metrics'])
                path = root / f'{mode}_stability.json'
                cli = subprocess.run([sys.executable, '-m', 'src.evaluation', '--stability',
                                      *report['run_dirs'], '--report', str(path)],
                                     capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(cli.stdout), json.loads(path.read_text()))
                self.assertEqual(evaluate_stability(report['run_dirs']), report['stability'])
                self.assertEqual(report['stability']['tasks']['joint']['stable_fraction'], 1)


if __name__ == '__main__':
    unittest.main()
