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
    def test_severity_digit_derivation_strict_failures_and_persistence(self):
        raw = ['0', ' 1\n', '2', '3', 'severity 2', '"2"', '4', '2\n3', '']
        class Digits:
            def generate_batch(self, paths, prompt):
                return [raw[int(path)] for path in paths]
        with tempfile.TemporaryDirectory() as folder:
            results = run_inference(Digits(), map(str, range(len(raw))),
                                    response_format='severity_digit', batch_size=3,
                                    output_dir=folder, show_progress=False)
            self.assertEqual([r.prediction for r in results[:4]], [
                {'noise': '0', 'severity': '0', 'route': '0'},
                {'noise': '1', 'severity': '1', 'route': '0'},
                {'noise': '1', 'severity': '2', 'route': '1'},
                {'noise': '1', 'severity': '3', 'route': '1'},
            ])
            self.assertTrue(all(r.error and r.prediction is None for r in results[4:]))
            self.assertEqual([r.raw_output for r in results], raw)
            run = Path(folder) / 'experiment_001/run_001'
            self.assertEqual(load_results(run), results)
            manifest = json.loads((run / 'manifest.json').read_text())
            self.assertEqual(manifest['response_format'], 'severity_digit')
            self.assertEqual(manifest['failed_images'], 5)
        result = run_inference(lambda path, prompt: '3', ['a'], output_dir=None,
                               response_format='severity_digit', show_progress=False)[0]
        self.assertEqual(result.prediction['severity'], '3')
        for kwargs in ({'response_format': 'unknown'},
                       {'response_format': 'severity_digit', 'inference_mode': 'few_shot'}):
            with self.assertRaises(ValueError):
                run_inference(Digits(), [], output_dir=None, show_progress=False, **kwargs)

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


class AdapterTests(unittest.TestCase):
    def test_full_resolution_preserves_pixels_and_disables_resize(self):
        import torch
        import numpy as np
        from PIL import Image
        from types import SimpleNamespace
        from src.transformers_adapter import TransformersImageModel

        class Inputs(dict):
            def to(self, device):
                return self
        class Processor:
            tokenizer = SimpleNamespace(padding_side='right')
            image_processor = SimpleNamespace(patch_size=16, merge_size=2)
            def apply_chat_template(self, conversations, **kwargs):
                self.kwargs = kwargs
                im = conversations[0][0]['content'][0]['image']
                self.pixels = np.asarray(im).copy()
                w, h = im.size
                return Inputs(input_ids=torch.tensor([[1]]),
                              image_grid_thw=torch.tensor([[1, h // 16, w // 16]]))
            def batch_decode(self, *args, **kwargs):
                return [CLEAN]
        class Model:
            device = 'cpu'
            def eval(self):
                pass
            def generate(self, **kwargs):
                return torch.tensor([[1, 2]])
        processor = Processor()
        adapter = TransformersImageModel(Model(), processor, full_resolution=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'image.png'
            source = np.arange(65 * 47 * 3, dtype=np.uint8).reshape(65, 47, 3)
            Image.fromarray(source).save(path)
            adapter.generate(path, 'prompt')
        self.assertEqual(processor.pixels.shape, (96, 64, 3))
        np.testing.assert_array_equal(processor.pixels[:65, :47], source)
        np.testing.assert_array_equal(processor.pixels, np.pad(source, ((0, 31), (0, 17), (0, 0)), mode='edge'))
        self.assertFalse(processor.kwargs['processor_kwargs']['images_kwargs']['do_resize'])
        self.assertEqual(adapter.inference_config['image_views'], 'full_resolution')
        with self.assertRaises(ValueError):
            TransformersImageModel(Model(), processor, full_resolution=True, detail_crop_size=512)

    def test_native_detail_pixels_orientation_and_few_shot_views(self):
        import torch
        import numpy as np
        from PIL import Image, ImageOps
        from types import SimpleNamespace
        from src.transformers_adapter import TransformersImageModel
        from src.image_quality_pipeline import FewShotExample

        class Inputs(dict):
            def to(self, device):
                return self
        class Processor:
            tokenizer = SimpleNamespace(padding_side='right')
            image_processor = SimpleNamespace(size={'longest_edge': 1048576, 'shortest_edge': 65536})
            def apply_chat_template(self, conversations, **kwargs):
                self.views = [[(item['image'].size, item['image'].tobytes())
                               for item in turn['content'] if item['type'] == 'image']
                              for turn in conversations[0] if turn['role'] == 'user']
                return Inputs(input_ids=torch.tensor([[1, 2]]))
            def batch_decode(self, tokens, **kwargs):
                return [CLEAN]
        class Model:
            device = 'cpu'
            def eval(self):
                pass
            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 3]])

        processor = Processor()
        adapter = TransformersImageModel(Model(), processor, detail_crop_size=512)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'oriented.png'
            pixels = np.arange(768 * 640 * 3, dtype=np.uint8).reshape(768, 640, 3)
            im = Image.fromarray(pixels)
            exif = im.getexif()
            exif[274] = 6
            im.save(path, exif=exif)
            with Image.open(path) as source:
                expected = ImageOps.exif_transpose(source).convert('RGB')
            width, height = expected.size
            crop = expected.crop(((width - 512) // 2, (height - 512) // 2,
                                  (width - 512) // 2 + 512, (height - 512) // 2 + 512))
            expected_views = [(expected.size, expected.tobytes()), (crop.size, crop.tobytes())]
            for examples in [(), (FewShotExample(str(path), 'ref',
                                  {'noise': '0', 'severity': '0', 'route': '0'}),)]:
                adapter.generate(path, 'prompt', examples=examples)
                self.assertEqual(processor.views, [expected_views] * (len(examples) + 1))
            small = Path(folder) / 'small.png'
            Image.new('RGB', (256, 256)).save(small)
            with self.assertRaisesRegex(ValueError, 'too small'):
                adapter.generate(small, 'prompt')
        self.assertEqual(adapter.inference_config['detail_crop_size'], 512)
        for invalid in [True, 128, 513, 512.0]:
            with self.assertRaises(ValueError):
                TransformersImageModel(Model(), processor, detail_crop_size=invalid)
        processor.image_processor = SimpleNamespace(size={'longest_edge': 65536, 'shortest_edge': 65536})
        with self.assertRaisesRegex(ValueError, 'resize the native detail crop'):
            TransformersImageModel(Model(), processor, detail_crop_size=512)

    def test_batched_padding_prompt_trimming_and_order(self):
        import torch
        from PIL import Image
        from types import SimpleNamespace
        from src.transformers_adapter import TransformersImageModel

        class Inputs(dict):
            def to(self, device):
                return self
        class Processor:
            tokenizer = SimpleNamespace(padding_side='right')
            def apply_chat_template(self, conversations, **kwargs):
                self.conversations = conversations
                self.kwargs = kwargs
                return Inputs(input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]))
            def batch_decode(self, tokens, **kwargs):
                self.tokens = tokens.tolist()
                return ['  ' + CLEAN, NOISY + '  ']
        class Model:
            device = 'cpu'
            def eval(self):
                self.evaluation = True
            def generate(self, **kwargs):
                self.kwargs = kwargs
                assert not torch.is_grad_enabled()
                return torch.tensor([[0, 1, 2, 6, 7], [3, 4, 5, 8, 9]])
        processor, model = Processor(), Model()
        adapter = TransformersImageModel(model, processor)
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / f'{i}.png' for i in range(2)]
            for path in paths:
                Image.new('RGB', (8, 8)).save(path)
            self.assertEqual(adapter.generate_batch(paths, 'prompt'), [CLEAN, NOISY])
            self.assertTrue(all(len(c) == 1 for c in processor.conversations))
            from src.image_quality_pipeline import FewShotExample
            examples = []
            for severity in ['0', '1', '2', '3']:
                reference_path = Path(folder) / f'reference_{severity}.png'
                Image.new('RGB', (8, 8)).save(reference_path)
                prediction = {'noise': '0' if severity == '0' else '1', 'severity': severity,
                              'route': '0' if severity in ['0', '1'] else '1'}
                examples.append(FewShotExample(str(reference_path), 'reference', prediction))
            self.assertEqual(adapter.generate_batch(paths, 'prompt', examples=examples), [CLEAN, NOISY])
            for conversation in processor.conversations:
                self.assertEqual([turn['role'] for turn in conversation], ['user', 'assistant'] * 4 + ['user'])
                self.assertEqual(len([turn for turn in conversation if turn['role'] == 'user']), 5)
                for index, example in enumerate(examples):
                    self.assertEqual(json.loads(conversation[2 * index + 1]['content'][0]['text']), example.prediction)
                self.assertIn('Target image', conversation[-1]['content'][1]['text'])
        self.assertTrue(model.evaluation)
        self.assertEqual(processor.tokenizer.padding_side, 'left')
        self.assertTrue(processor.kwargs['processor_kwargs']['padding'])
        self.assertEqual(processor.tokens, [[6, 7], [8, 9]])
        self.assertEqual(len(processor.conversations), 2)
        self.assertFalse(model.kwargs['do_sample'])
        self.assertEqual(adapter.generate_batch([], 'prompt'), [])


if __name__ == '__main__':
    unittest.main()
