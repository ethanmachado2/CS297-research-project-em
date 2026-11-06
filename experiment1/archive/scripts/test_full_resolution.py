"""Compare resized and pixel-preserving full-image inference on one subject."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HOME'] = str(ROOT / 'hf_cache')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', default='35')
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/full_resolution_test')
    args = parser.parse_args()

    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    from src.image_quality_pipeline import build_dataset, run_inference, _write_json
    from src.evaluation import evaluate_saved_run
    from src.transformers_adapter import TransformersImageModel

    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable in this execution context; GPU access is needed for this test.')
    if args.subject == '0':
        parser.error('Subject 0 is reserved for calibration')
    frame = build_dataset(ROOT / 'data/files')
    frame = frame.loc[frame.subject_id == args.subject].sort_values('severity')
    if set(frame.severity) != {'0', '1', '2', '3'} or len(frame) != 4:
        parser.error('Subject must have exactly four severity levels')
    if args.output_dir.exists():
        raise FileExistsError(f'Use a fresh output directory: {args.output_dir}')
    args.output_dir.mkdir(parents=True)
    snapshot_root = ROOT / 'hf_cache/hub/models--Qwen--Qwen3.5-0.8B'
    revision = (snapshot_root / 'refs/main').read_text().strip()
    snapshot = snapshot_root / 'snapshots' / revision
    summary = {'status': 'running', 'subject': args.subject, 'device': args.device,
               'model_revision': revision, 'batch_size': 1, 'mode': 'zero_shot',
               'prompt': 'revised grain-contrast prompt', 'experiments': {}}
    _write_json(args.output_dir / 'comparison.json', summary)
    start = perf_counter()
    try:
        print('Loading cached model...', flush=True)
        model = AutoModelForMultimodalLM.from_pretrained(snapshot, local_files_only=True).to(args.device)
        for experiment, full_resolution in enumerate([False, True], 1):
            name = 'full_resolution' if full_resolution else 'resized_262144'
            processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
            processor.image_processor.size['longest_edge'] = 262144
            adapter = TransformersImageModel(model, processor, max_new_tokens=64,
                                              full_resolution=full_resolution)
            print(f'Running {name}: four severity levels, one image per batch', flush=True)
            geometry = []
            for path in frame.file_path:
                with Image.open(path) as source:
                    im = ImageOps.exif_transpose(source)
                    w, h = im.size
                    geometry.append({'path': path, 'source_size': [w, h],
                                     'padded_full_size': [w + (-w) % 32, h + (-h) % 32]})
            rows = run_inference(adapter, frame.file_path, labels=frame, batch_size=1,
                                 output_dir=args.output_dir, experiment_number=experiment,
                                 model_name='Qwen/Qwen3.5-0.8B', show_progress=True)
            run = args.output_dir / f'experiment_{experiment:03d}/run_001'
            summary['experiments'][name] = {
                'run_dir': str(run), 'config': adapter.inference_config, 'geometry': geometry,
                'metrics': evaluate_saved_run(run, save=False),
                'results': [{'severity_truth': truth, 'raw_output': row.raw_output,
                             'prediction': row.prediction, 'error': row.error}
                            for truth, row in zip(frame.severity, rows)],
            }
            _write_json(args.output_dir / 'comparison.json', summary)
            if full_resolution and any(row.error and ('memory' in row.error.lower()) for row in rows):
                summary['status'] = 'completed_with_memory_errors'
            if args.device == 'mps':
                torch.mps.empty_cache()
        if summary['status'] == 'running':
            summary['status'] = 'completed'
    except BaseException as exc:
        summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        summary['elapsed_seconds'] = perf_counter() - start
        _write_json(args.output_dir / 'comparison.json', summary)
        print(f'Evidence: {args.output_dir / "comparison.json"}', flush=True)


if __name__ == '__main__':
    main()
