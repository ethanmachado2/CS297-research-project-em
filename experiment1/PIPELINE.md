# Before-resize-cap experiment workflow

The active project contains three notebooks:

- [Zero-shot](notebooks/cs297-experiment1-em-local-before-resize-cap.ipynb): original prompt and configuration, 200 evaluation images.
- [Few-shot](notebooks/cs297-experiment1-em-local-before-resize-cap-few-shot.ipynb): original base prompt plus calibration guidance, subject `0`'s four labeled reference images, and 196 evaluation targets excluding that subject.
- [TRM supervised](notebooks/cs297-experiment1-em-local-trm.ipynb): a jointly trained image encoder and recursive model, with 140 training / 20 validation / 40 held-out images grouped by person. Setup and results are documented below.

Select the project `.venv` kernel and restart the kernel before Run All. The two Qwen notebooks use the cached Qwen/Qwen3.5-0.8B model, the processor's default resizing, one target at a time, one pass, greedy generation and at most 128 new tokens. Few-shot conversations contain four references plus the target. No custom pixel cap or detail crop is applied in those notebooks.

`src/image_quality_pipeline.py` supplies dataset labels, reference holdout, prompt guidance, strict JSON validation, progress reporting and per-image checkpoints. `src/evaluation.py` computes metrics from saved predictions. Inference adapters are defined within the notebooks; the previous shared resizing/crop adapter is archived.

## Saved results

- `outputs/original_configuration_runs/`: zero-shot sessions.
- `outputs/original_configuration_few_shot_runs/`: few-shot sessions.
- `outputs/recovered_history/`: recovered notebook, original pipeline source and historical aggregate metrics, retained as provenance.

Each new session contains `experiment_001/run_001/` with its manifest, labels, individual `batch_*.json` predictions and metrics. The final notebook section exports prediction, failure, evaluation-summary and class-coverage CSVs. It can run offline against a saved `RUN_DIR` without loading the model. Incomplete sessions are identified as partial results.

Accuracy, precision, recall, F1, confusion matrices and severity MAE score valid predictions. Failures and coverage are reported separately; correct/attempted also counts failures as unsuccessful attempts. References are excluded from few-shot metrics. Compare both modes on the same 196 targets when making a matched comparison.

## Verification and archive

Run the active shared-code tests with:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Earlier variants and their outputs, scripts and adapter tests are under [archive/](archive/README.md). The archive preserves original directory structure and records file hashes in `archive/archive_manifest.json`. The previous full workflow documentation is retained at `archive/PIPELINE.md`. Shared pipeline/evaluation modules remain intact, including their existing auxiliary capabilities, to preserve runtime behavior.

## TRM supervised pipeline

The [implementation specifications](plans/README.md) describe the first TRM experiment. The default architecture has **556,836 trainable parameters**: encoder 31,904, shared reasoner 524,288, classifier 516, and readout vector 128. It consumes nine native-scale 128×128 RGB patches from each image. A shared three-layer CNN produces nine width-128 vectors; a learned readout makes ten sequence positions. The official two attention/SwiGLU reasoning layers refine answer and working states with `H=2`, `L=2`, and four outer steps (24 reasoner calls / 48 block executions). Only the final answer readout predicts severity; noise and route follow the existing label mapping.

Official code is vendored at commit `c01103738605ba39d1430519b1ee0c62f4c707f8` under [vendor/tiny_recursive_models](vendor/tiny_recursive_models/PROVENANCE.json), including its MIT license and file hashes. Only import namespaces changed in those source files. The local image model omits token/puzzle embeddings, token output, and halting heads; it uses a smaller width and recursion budget, fixed-step supervision, float32, ordinary cross-entropy, and AdamW. This is an adaptation of the official TRM, not a reproduction of its puzzle benchmark recipe.

The original experiment environment used Python 3.14.8 and PyTorch 2.14.1. For a fresh environment, follow the Python 3.13/3.14 setup in [README.md](README.md). The main requirements now include the official modules' additional dependencies:

```sh
.venv/bin/python -m pip install -r requirements.txt
```

Then open the TRM notebook with the `.venv` kernel and Restart / Run All. Its visible config loads [configs/trm_experiment1.json](configs/trm_experiment1.json). Each session saves configuration, person split, image hashes, environment, parameter counts, per-epoch history, and a CPU-portable `checkpoints/best.pt`. The trainer supervises every outer refinement and selects the earliest best validation loss, with patience 15 and a maximum of 100 epochs. It does not feed test pixels into training or checkpoint selection. The final held-out pass uses `inference_mode="supervised"`, one image per batch, the existing runner/evaluator, and the original Qwen prompt as unused task metadata.

For a command-line training and held-out evaluation run from the project root:

```sh
OMP_NUM_THREADS=2 .venv/bin/python -m src.trm_training --session outputs/trm_supervised_runs/my_new_session --evaluate
```

Use a new session path each time. `auto` chooses CUDA, then MPS, then CPU. Explicit unavailable devices fail. Training resume is not implemented. Interruptions/failures preserve the best completed checkpoint and history and propagate training status into later inference reports. For evaluation only, set `TRAIN_MODEL=False` and `CHECKPOINT_PATH` in the notebook: its original configuration and verified split are reused. A moved project is supported through relative dataset paths and file hashes.

Reusable components are `src/trm_data.py`, `src/trm_model.py`, `src/trm_training.py`, and `src/trm_adapter.py`. `src/trm_reporting.py` exports the same four CSVs as the Qwen notebooks without importing neural models. Run the final offline notebook cell independently after setting `RUN_DIR`; partial runs retain failure and unattempted counts.

The optional `TRM_CONFIG_PATH`, `TRM_DATA_PROJECT_ROOT`, `TRM_OUTPUT_ROOT`, `TRM_TRAIN_MODEL`, and `TRM_CHECKPOINT_PATH` environment variables support automated smoke runs. Leave them unset for ordinary real-data notebook use. The reproducible notebook smoke command creates synthetic data and uses fresh kernels for training, checkpoint-only inference, and offline export:

```sh
.venv/bin/python scripts/validate_trm_notebook.py --output outputs/trm_validation/new_smoke_session
TRM_OVERFIT_DIAGNOSTIC=1 .venv/bin/python -m unittest discover -s tests -p test_trm_training.py -k tiny_four_class -v
```

Default unit tests use synthetic CPU fixtures and no downloads. MPS/CUDA smoke tests run when those devices are available; the overfit diagnostic is opt-in. Notebook execution needs permission to open local Jupyter kernel sockets. Measured implementation acceptance results are recorded in [plans/04-validation-results.md](plans/04-validation-results.md).

## Diagnosing TRM learning

The initial checkpoint's constant predictions were traced through the native
patches, CNN feature maps, projected tokens, each recursive update, classifier,
and gradients. [The diagnosis](plans/05-learning-diagnosis.md) includes the measured
results and reproduction commands. `src/trm_diagnostics.py` supplies reusable
tracing and frozen-feature probes; `scripts/diagnose_trm.py` traces only the saved
training/validation membership, and `scripts/diagnose_trm_controls.py` runs
separate small training-subset interventions. Both scripts require new output
directories and preserve the original experiment and production configuration.
