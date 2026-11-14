# 02 — Training and Experiment 1 notebook specification

Depends on: [data and model](01-data-and-model.md). Validation: [test specification](03-tests-and-acceptance.md).

## Configuration and environment

Create `configs/trm_experiment1.json` and load it in a visible notebook configuration cell. Resolve and save the full effective configuration before training.

| Setting | Initial value |
| --- | --- |
| Split / model seed | `42` / `42` |
| Split people | `35 / 5 / 10` |
| Patch grid / patch size | `3×3 / 128` |
| Hidden width / reasoning layers / heads | `128 / 2 / 4` |
| H cycles / L cycles / refinement steps | `2 / 2 / 4` |
| Optimizer | PyTorch AdamW, learning rate `0.0003`, weight decay `0.0001` |
| Training batch size | `8`; retain the final smaller batch |
| Maximum epochs / early-stop patience | `100 / 15` |
| Loss / gradient clipping | Severity cross-entropy / global norm `1.0` |
| Device / dtype | `auto` / float32 |
| Loader workers | `0` initially for local notebook compatibility |
| Evaluation | Batch size `1`, one pass, final refinement prediction |

These settings are initial hypotheses. Accuracy is not guaranteed and is not an acceptance gate. Seed Python, NumPy, PyTorch, and loader shuffling; record device/library versions without promising identical floating-point training across devices.

`auto` selects CUDA when available, otherwise MPS, otherwise CPU; print and save the choice. Explicit unavailable devices fail clearly. Do not silently move training to CPU after a device error. Avoid CUDA-specific optimizers, mandatory mixed precision, distributed training, `torch.compile`, or external experiment trackers in this first version.

The original plan proposed a separate TRM requirements file. Dependencies are now
consolidated in `requirements.txt`; no TRM-specific requirements entry point remains.
Local notebooks keep installation in documentation. The separate Colab Enterprise
notebooks include an explicit bootstrap cell and preserve the managed Torch/Jupyter
stack through `scripts/install_cloud_dependencies.py`.

## Trainer

Implement reusable functions in `src/trm_training.py`, called by the notebook:

1. Validate the dataset and persist its split manifest before constructing loaders.
2. Initialize encoder, reasoner, and classifier from scratch; train them jointly.
3. For each minibatch, reset recurrent states. For each of four outer steps, recompute the encoded input with current weights, clear gradients, refine once, calculate severity cross-entropy, backpropagate, clip gradients, and update AdamW. Pass only detached states into the next step. This gives four optimizer updates per minibatch; record update counts explicitly.
4. Reject non-finite loss/gradients with a failed training manifest. Do not cache or detach learned image embeddings during training.
5. After each epoch, run the complete validation partition under `eval()` and inference mode with the same recursion budget. Compute loss from final logits and severity metrics; aggregate by image, including the final smaller batch.
6. Select the checkpoint with the lowest validation severity loss; ties keep the earliest checkpoint. An improvement is a decrease greater than `1e-6`. Stop after 15 consecutive epochs without improvement or at epoch 100.
7. Reload `best.pt` for the single held-out test pass. Do not use test labels or test performance to choose a checkpoint or settings. Do not retrain on validation after selection in version 1.

Write an atomic best-checkpoint file containing model state, architecture and preprocessing configs, split/dataset digest, upstream SHA, epoch, validation selection metric, and seeds. Keep saved model tensors portable to CPU loading. A new process must be able to reconstruct inference using only this checkpoint plus the image files. Training resume is outside version 1; an interrupted run retains its last completed best checkpoint and history.

Use `training_manifest.json` with `running`, `completed`, `interrupted`, or `failed` status and start/end times. Record effective config, environment, train/validation counts, parameter counts, elapsed time, checkpoint hash, and best epoch. Append or atomically rewrite `training_history.csv` after every completed epoch with mean training step loss, validation loss, severity accuracy/macro-F1, and optimizer-update count. Early stopping is normal completion; interruption is not a completed training result.

## Shared inference integration

Add `TRMImageModel` in `src/trm_adapter.py` with:

```python
adapter = TRMImageModel.from_checkpoint(checkpoint_path, device="auto")
adapter.generate(image_path, prompt) -> str
adapter.inference_config -> dict
```

`generate` loads patches using checkpoint preprocessing, runs a fresh-state final prediction in `eval()`/inference mode, takes argmax over four logits, and returns exactly one JSON object with string keys/values `noise`, `severity`, `route`. The mapping is the existing `0→0/0`, `1→1/0`, `2→1/1`, `3→1/1` for noise/route. Raise on unreadable images or non-finite logits; let the shared runner record those failures. Do not return a fabricated class after an error.

The `prompt` argument maintains adapter compatibility; it is not tokenized or consumed by TRM. Record `uses_prompt=false` and `training_regime="supervised"` in `inference_config`. Also save checkpoint identity/hash, upstream SHA, device/dtype, preprocessing/model configs, split digest, parameter counts, and refinement budget. Time image loading, encoding, and recursion inside each adapter call so existing batch timings include all three.

Make a narrow backward-compatible extension to `src/image_quality_pipeline.py`:

- Add `"supervised"` to `INFERENCE_MODES`; reject nonempty `examples` in this mode before inference or run-folder creation.
- Keep the default `"zero_shot"`, few-shot guidance/holdout behavior, response validation, and existing artifact schema unchanged.
- Use `response_format="json"` for TRM. Preserve the current severity-digit restrictions; that auxiliary mode need not support supervised runs.
- Persist `inference_mode="supervised"` in the existing manifest and batch records.

Call `run_inference` with **test rows only**, `labels=test_frame`, no examples, batch size one, and the original Experiment 1 prompt stored as task metadata. Do not hand the runner the full 200-image label frame: label coverage must refer to the 40 planned test images.

## Notebook workbook

Create `notebooks/cs297-experiment1-em-local-trm.ipynb`, retaining the existing title/author style and these sections:

1. Experiment description and explicit supervised-versus-zero-shot distinction.
2. Setup, project-root discovery, configuration, seed, environment/device display.
3. Dataset validation, saved group split, class counts, and patch preview.
4. Encoder/TRM initialization and parameter counts.
5. Training with progress, validation history, and checkpoint selection.
6. Reload best checkpoint and expose the inference adapter/config.
7. Test inference through `run_inference`, with per-image saved predictions.
8. Offline review and CSV export, runnable from a saved `RUN_DIR` without loading TRM, its checkpoint, Qwen, or a GPU.

Fresh setup creates a new UTC timestamp plus UUID session, following existing notebooks. Restart-and-Run-All performs training and then testing. Provide an explicit `TRAIN_MODEL=False` plus `CHECKPOINT_PATH` path for evaluating an existing checkpoint; this must load its original split and preprocessing rather than create a new split. Evaluation from an interrupted training checkpoint must retain that provenance in the report.

Keep the existing offline report columns and exports: `predictions.csv`, `failures.csv`, `evaluation_summary.csv`, `coverage_by_severity.csv`. Display confusion matrices with true rows/predicted columns, severity MAE, valid accuracy, correct/attempted, and correct/planned. Retain the existing partial-run warning based on manifest status and counts.

## Artifact layout

```text
outputs/trm_supervised_runs/<SESSION_ID>/
    config.json
    split.json
    training_manifest.json
    training_history.csv
    checkpoints/best.pt
    experiment_001/run_001/
        manifest.json
        labels.json
        batch_00001.json ... batch_00040.json
        metrics.json
        predictions.csv
        failures.csv
        evaluation_summary.csv
        coverage_by_severity.csv
```

Reference training artifacts from the inference config. A checkpoint-only evaluation session writes its own config/split/provenance references without inventing a new training history. Preserve the runner's refusal to overwrite an existing explicit run number. Keep large generated weights out of source control if a Git repository is used; do not delete or relocate existing outputs.

## Results and optional matched Qwen report

Report noise/severity/route accuracy, precision, recall, macro-F1, confusion matrices, severity MAE, coverage/failures, training duration, end-to-end inference time, parameter counts, and the recursive budget. Explain that patch sampling and supervised training differ from Qwen's original inference setup. Routing labels describe whether to denoise; no denoiser executes here.

If comparing Qwen, filter its saved results to exactly these test images and use the same metrics. Normalize historical absolute paths through project-relative paths; require a unique complete match and matching dataset hashes where available. Preserve invalid Qwen outputs as failures and show partial coverage when predictions are missing. Do not compare TRM's 40-image score with Qwen's full 200-image aggregate as a matched comparison. This optional report must not block the TRM experiment.
