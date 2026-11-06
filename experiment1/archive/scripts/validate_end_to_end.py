"""Run real cached-model inference and verify saved evaluation round trips.

This is an opt-in integration check, separate from the fast unit-test suite.
Use --max-pixels to select an explicit processor resize budget; omitted means
use the cached model's defaults. --detail-crop-size adds a native center view
to every reference and target. No images or model weights are downloaded.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = str(ROOT / 'hf_cache')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'end_to_end')
    parser.add_argument('--max-pixels', type=int)
    parser.add_argument('--detail-crop-size', type=int, help='Native center crop edge; multiple of 32, at least 256')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--subjects', type=int, help='Limit evaluation subjects for a smoke test')
    parser.add_argument('--modes', nargs='+', choices=['zero_shot', 'few_shot'], default=['zero_shot', 'few_shot'])
    args = parser.parse_args()
    if args.subjects is not None and args.subjects < 1:
        parser.error('--subjects must be positive')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if len(set(args.modes)) != len(args.modes):
        parser.error('--modes must not contain duplicates')
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    from src.image_quality_pipeline import build_dataset, prepare_few_shot_split, load_results, _write_json
    from src.transformers_adapter import TransformersImageModel
    from src.evaluation import EvaluationConfig, run_stability_experiment, evaluate_saved_run, evaluate_stability

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / 'validation.json'
    if summary_path.exists():
        raise FileExistsError(f'Use a fresh output directory: {args.output_dir}')
    summary = {'status': 'running', 'python': sys.version, 'torch': torch.__version__,
               'transformers': transformers.__version__, 'checks': [], 'experiments': {}}
    started = perf_counter()
    try:
        df = build_dataset(ROOT / 'data' / 'files')
        evaluation, examples = prepare_few_shot_split(df, '0')
        if args.subjects:
            subjects = sorted(evaluation.subject_id.unique(), key=int)[:args.subjects]
            evaluation = evaluation.loc[evaluation.subject_id.isin(subjects)].copy()
        assert not evaluation.empty
        assert set(evaluation.severity) == {'0', '1', '2', '3'}
        assert '0' not in set(evaluation.subject_id)
        summary.update(dataset_images=len(df), evaluation_images=len(evaluation), passes=3,
                       batch_size=args.batch_size, subject_limit=args.subjects,
                       evaluation_paths=evaluation.file_path.tolist())
        model_path = ROOT / 'hf_cache' / 'hub' / 'models--Qwen--Qwen3.5-0.8B'
        revision = (model_path / 'refs' / 'main').read_text().strip()
        snapshot = model_path / 'snapshots' / revision
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        if args.max_pixels is not None:
            if args.max_pixels < processor.image_processor.size['shortest_edge']:
                raise ValueError('--max-pixels must be at least the processor minimum')
            processor.image_processor.size['longest_edge'] = args.max_pixels
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        summary.update(device=device, model_revision=revision, processor_size=dict(processor.image_processor.size))
        print(json.dumps(summary, indent=2), flush=True)
        print('Loading cached model...', flush=True)
        model = AutoModelForMultimodalLM.from_pretrained(snapshot, local_files_only=True).to(device)
        adapter = TransformersImageModel(model, processor, max_new_tokens=128,
                                         detail_crop_size=args.detail_crop_size)
        summary['detail_crop_size'] = args.detail_crop_size
        for experiment, mode in enumerate(args.modes, 1):
            print(f'Running {mode}: {len(evaluation)} images x 3 passes', flush=True)
            report = run_stability_experiment(
                adapter, evaluation.file_path, labels=evaluation,
                config=EvaluationConfig(args.output_dir, experiment), batch_size=args.batch_size,
                inference_mode=mode, examples=examples, model_name='Qwen/Qwen3.5-0.8B',
                show_progress=True,
            )
            summary['experiments'][mode] = {'report_path': report['report_path'], 'run_dirs': report['run_dirs']}
            _write_json(summary_path, summary)
            for run_dir, metrics in zip(report['run_dirs'], report['metrics']):
                run = Path(run_dir)
                manifest = json.loads((run / 'manifest.json').read_text())
                results = load_results(run)
                assert manifest['status'] == 'completed'
                assert manifest['total_images'] == manifest['completed_images'] == len(evaluation)
                assert len(results) == len(evaluation)
                assert {r.image_path for r in results} == set(evaluation.file_path)
                assert all(Path(r.image_path).stem != '0' for r in results)
                assert evaluate_saved_run(run, save=False) == metrics == json.loads((run / 'metrics.json').read_text())
                for task in ('noise', 'severity', 'route'):
                    task_metrics = metrics[task]
                    matrix = task_metrics['confusion_matrix']
                    assert sum(map(sum, matrix)) == task_metrics['evaluated']
                    assert sum(matrix[i][i] for i in range(len(matrix))) == task_metrics['correct']
                cli = subprocess.run([sys.executable, '-m', 'src.evaluation', run_dir], cwd=ROOT,
                                     capture_output=True, text=True, check=True)
                assert json.loads(cli.stdout)[run_dir] == metrics
                # A completed run with zero valid responses is not a successful integration test.
                assert metrics['severity']['evaluated'] == len(evaluation), f'Invalid or missing responses in {run_dir}' 
            assert evaluate_stability(report['run_dirs']) == report['stability']
            cli_report = Path(report['report_path']).parent / 'cli_stability.json'
            cli = subprocess.run([sys.executable, '-m', 'src.evaluation', '--stability', *report['run_dirs'],
                                  '--report', str(cli_report)], cwd=ROOT, capture_output=True, text=True, check=True)
            assert json.loads(cli.stdout) == json.loads(cli_report.read_text()) == report['stability']
            summary['experiments'][mode].update(
                failed_images=[m['severity']['failed'] for m in report['metrics']],
                coverage=[m['severity']['label_coverage'] for m in report['metrics']],
                routing_accuracy=[m['route']['accuracy'] for m in report['metrics']],
                severity_mae=[m['severity']['mae'] for m in report['metrics']],
                routing_stability={k: v for k, v in report['stability']['tasks']['route'].items() if k != 'per_image'},
            )
            summary['checks'].append(f'{mode}: real inference, checkpoint/reload, metrics, CLI, stability')
            _write_json(summary_path, summary)
        summary['status'] = 'passed'
    except BaseException as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        summary['elapsed_seconds'] = perf_counter() - started
        _write_json(summary_path, summary)
        print(f'Validation evidence: {summary_path}', flush=True)


if __name__ == '__main__':
    main()
