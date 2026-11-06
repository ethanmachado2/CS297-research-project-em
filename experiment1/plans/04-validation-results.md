# 04 — Implementation validation and initial results

The pipeline is implemented. This record describes executed checks, not a target accuracy claim. Existing Qwen notebooks and outputs were preserved. The shared runner adds only the supervised mode and its rejection of reference examples.

## Validation performed

- Default discovery: **60 tests, 57 passed, 3 skipped**, in 5.298 seconds. The skipped cases were CUDA (unavailable), MPS (hidden inside the sandbox), and the opt-in learnability diagnostic. [Complete unit-test log](../outputs/trm_validation/unit_tests.txt).
- MPS: the model test suite was repeated outside the sandbox; **8 passed, 1 CUDA skip**, including a real MPS forward/backward step and device-state check. CUDA was not tested; no GCP resources were provisioned.
- Synthetic learnability: the opt-in diagnostic passed. On four constant-color synthetic classes, 150 iterations of two supervised refinements reduced training loss from **1.58878 to 0.06636**, with **100% training accuracy**. This is a learning-path check, not held-out biometric evidence.
- Notebook: fresh kernels executed synthetic training from the project root, checkpoint-only evaluation from `notebooks/`, and the actual offline export cell with neural-model imports forbidden. Both inference sessions produced four valid predictions and all four CSVs. [Notebook smoke summary](../outputs/trm_validation/notebook_smoke_final/summary.json).
- Notebook source passes `nbformat` validation and has cleared outputs. Checkpoint reconstruction was also exercised in a fresh Python process. Model-free reporting was tested without importing torch, transformers, the encoder, adapter, or trainer.
- Tests cover person-disjoint splits and relocation/content checks; exact patch coordinates/color conversion; gradient flow through the real encoder/reasoner/classifier; detached outer states; 24 shared-reasoner / 48 block calls; reset between images; validation-only checkpoint selection and ties; batch remainders; non-finite loss/gradients; interruptions; per-image failures; partial coverage; strict JSON; and existing zero/few-shot regressions.

Reproduce the unit suite and optional diagnostic using the commands in [PIPELINE.md](../PIPELINE.md). Notebook smoke execution requires local Jupyter sockets. The first sandbox attempt was blocked by that restriction; the final successful run used normal local kernels outside the sandbox. A cleanup issue in the smoke script's temporary import guard was fixed and the fresh-kernel smoke was repeated.

## Real Experiment 1 run

Artifacts: [session](../outputs/trm_supervised_runs/20261024_implementation_validation/training_manifest.json), [configuration](../outputs/trm_supervised_runs/20261024_implementation_validation/config.json), [split](../outputs/trm_supervised_runs/20261024_implementation_validation/split.json), [training history](../outputs/trm_supervised_runs/20261024_implementation_validation/training_history.csv), [best checkpoint](../outputs/trm_supervised_runs/20261024_implementation_validation/checkpoints/best.pt), [test report](../outputs/trm_supervised_runs/20261024_implementation_validation/experiment_001/run_001/evaluation_summary.csv), [predictions](../outputs/trm_supervised_runs/20261024_implementation_validation/experiment_001/run_001/predictions.csv).

- All 200 files validated: 50 distinct people, one image per person per severity. Person groups are disjoint: **140 train / 20 validation / 40 test images**, with equal severity support in each partition.
- Official upstream commit: `c01103738605ba39d1430519b1ee0c62f4c707f8`. Vendored source hashes and import-only modifications are recorded in [provenance](../vendor/tiny_recursive_models/PROVENANCE.json); the MIT license is retained.
- Float32 on CPU, Python 3.14.8, PyTorch 2.14.1, two CPU threads via `OMP_NUM_THREADS=2`. The requested device was `auto`; the sandbox exposed CPU. The real training run used the documented defaults without tuning after the test pass.
- Parameters: **556,836 total** = 31,904 encoder + 524,288 reasoner + 516 classifier + 128 readout. Width 128, 4 heads, 2 shared layers, H=2, L=2, K=4.
- Completed **23 epochs / 1656 optimizer updates**, stopping normally after 15 epochs without validation improvement. Selected epoch **8**, validation cross-entropy **1.387394**. Training duration **523.256 seconds** (about 8.72 minutes).
- Held-out inference: **40/40 attempted and valid, zero failures, 100% planned coverage**, **2.979 seconds** end to end through the runner (image loading, encoding, recursion, and saved-run overhead; model initialization and split hashing excluded).

| Task | Correct / planned | Accuracy | Precision | Recall | Macro-F1 | Severity MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Noise | 30 / 40 | 0.750 | 0.750 | 1.000 | 0.428571 | — |
| Severity | 10 / 40 | 0.250 | 0.0625 | 0.250 | 0.100000 | 1.000 |
| Route | 20 / 40 | 0.500 | 0.000 | 0.000 | 0.333333 | — |

Precision/recall use the existing evaluator: positive class 1 for noise/route and macro averaging for severity. All metrics use valid predictions; here valid, attempted, and planned denominators are each 40.

**The selected checkpoint predicts severity 1 (weak) for all 40 test images.** Severity's confusion matrix is `[[0,10,0,0],[0,10,0,0],[0,10,0,0],[0,10,0,0]]`. Consequently the derived noise/route accuracies reflect constant predictions. This initial default configuration has not learned useful severity discrimination on the real dataset. The successful synthetic overfit diagnostic shows that gradients and updates work; it does not establish real-data generalization. No settings were changed using held-out results, and no improvement over Qwen is claimed. A matched historical Qwen comparison remains optional and was not run.

Machine-readable evidence: [validation summary](../outputs/trm_validation/validation_summary.json). Best checkpoint SHA-256: `445d6c8a510fba648b729005aa2232cd5b3d962a4e01f5cfeaa7718fd207b4c3`.
