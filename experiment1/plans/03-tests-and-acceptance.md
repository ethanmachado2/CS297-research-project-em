# 03 — Test cases and acceptance

Test the implementations in [01](01-data-and-model.md) and [02](02-training-and-notebook.md). This file retains the agreed acceptance criteria; executed results are in [04](04-validation-results.md).

## Test conventions

Use the existing `unittest` discovery command:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Substitute the documented TRM environment interpreter if separate. Default tests use CPU, temporary directories, and synthetic images; no internet, model downloads, paid resources, or full training runs. Use small model configs where testing learning mechanics. Skip accelerator tests only when that accelerator is unavailable, with an explicit skip reason.

## Data and preprocessing — `tests/test_trm_data.py`

| ID | Case | Expected result |
| --- | --- | --- |
| D1 | Synthetic frame with 50 distinct people and all four severities | Exactly 35/5/10 people and 140/20/40 images; 35/5/10 examples per severity; pairwise disjoint people and paths. |
| D2 | Reorder input rows, preserving IDs, and use the same seed | Identical assigned ID lists. Saved manifests reload the same membership; row order follows the specified source ordering. |
| D3 | Missing severity, duplicate person/severity, duplicate image path, missing file, invalid label | Clear validation error before training. |
| D4 | Round-trip manifest, then move identical files to another temporary root | Same membership and hashes; no absolute local-path dependency. Mutating an image or introducing overlapping partitions is rejected. |
| D5 | Coordinate-coded RGB image with manually calculated crop bounds | Nine patches in row-major order with the specified coordinates, channel order, float32 type, and `[0,1]` scaling. |
| D6 | Grayscale/RGBA inputs; small image; corrupt image | Supported images convert to RGB; small/corrupt images fail explicitly. Rotation is not applied under the default EXIF setting. |
| D7 | Same pixels at different filenames/severity folders | Identical patch tensors; paths, labels, and metadata cannot change the encoded input. |

## Encoder and recurrence — `tests/test_trm_model.py`

| ID | Case | Expected result |
| --- | --- | --- |
| M1 | Batch size one and a larger batch through encoder and model | Encoder `[B,P,D]`; final logits `[B,4]`; finite outputs; correct configured dtype/device. |
| M2 | Supervise one training refinement on synthetic examples | Finite nonzero gradients reach encoder, shared reasoner, and classifier; an optimizer update changes each component. Initial no-gradient cycles do not cut the final encoder gradient path. |
| M3 | Execute two supervised outer steps with updates | No reused-autograd-graph error; carried states are detached and fresh image embeddings support gradients at each step. |
| M4 | Instrument reasoner calls for `H=2,L=2,K=4` | Exactly 24 shared-reasoner calls (48 reasoning-block executions with two layers); final-step logits are returned. The same layer parameters are reused. |
| M5 | Evaluate image A, then B, then A | Both A outputs agree within CPU float32 tolerance; B cannot contaminate A via recurrent state. |
| M6 | Save/reload state and config on CPU | Same input produces matching logits within tolerance; default parameter counts agree before/after load. |
| M7 | Zero/negative cycles or incompatible width/head configuration | Fail before tensor execution. |
| M8 | Inspect upstream provenance and instantiated model | Concrete SHA/license/source hashes are recorded; official reasoning classes are used; no instantiated puzzle embeddings, token LM head, or halting parameters are counted as useful classifier components. |

M4 checks recursion because it is the central scientific behavior, not just an implementation detail. Do not require different recursion budgets to improve accuracy in a unit test.

## Trainer and checkpoint — `tests/test_trm_training.py`

| ID | Case | Expected result |
| --- | --- | --- |
| T1 | Tiny CPU dataset, two epochs, batch remainder, multiple refinements | Finite losses; all training rows processed; expected optimizer-update count; history and portable checkpoint saved. |
| T2 | Instrument dataset accesses during training/selection | Only train/validation partitions are accessed; no test sample or label is used. Validation runs without gradient updates. |
| T3 | Controlled validation losses `[0.8,0.5,0.5,0.6]` | Earliest `0.5` checkpoint wins; ties/non-improvements increment patience; selection never depends on test accuracy. Test the `1e-6` improvement tolerance. |
| T4 | CPU reload followed by adapter inference, independent of trainer instance | Model and preprocessing reconstruct from the checkpoint; final logits match. Wrong split/config version produces a clear compatibility error. |
| T5 | Keyboard interruption after one completed epoch | Training manifest is interrupted; completed history and best checkpoint remain readable; no full-test completion claim is written. |
| T6 | Non-finite loss/gradient; unavailable explicitly selected device | Clear failure and persisted failed status where a session exists; no fabricated checkpoint or silent device switch. |
| T7 | Small learnability diagnostic, opt-in rather than ordinary discovery | On a fixed tiny synthetic four-class training set, training loss decreases substantially and the model can overfit. Record config and result; this is an engineering diagnostic, not held-out evidence. |

Use controlled metric fixtures for selection tests and real autograd/model operations for T1/M2/M3. Do not substitute mocks for the entire learning path.

## Runner and reporting — `tests/test_trm_integration.py`

| ID | Case | Expected result |
| --- | --- | --- |
| I1 | Controlled adapter logits selecting each of four severities | Exactly the existing three JSON keys with string values; all four mappings pass `validate_prediction`. No explanatory text. |
| I2 | Run a tiny real checkpoint through `run_inference` in supervised mode | Existing `experiment_001/run_001` artifacts reload; manifest/batches identify supervised mode; inference config includes checkpoint, upstream, preprocessing, and split identity. |
| I3 | Supply examples in supervised mode | Rejected before model invocation or output creation. Existing zero/few-shot behavior and tests remain unchanged. |
| I4 | Corrupt image or non-finite adapter logits among valid inputs | Per-image failure persisted; remaining images proceed; coverage and attempted counts reflect the failure. |
| I5 | Inference interruption, followed by offline evaluation | Only completed predictions are scored; unattempted images remain visible through planned counts; report labels the run incomplete. |
| I6 | Reuse an explicit run number | Existing results are not overwritten; `FileExistsError` follows current runner behavior. |
| I7 | Offline report with model/encoder imports disabled and no checkpoint present | Saved inference JSON is sufficient to produce all four existing CSV exports and confusion matrices. |
| I8 | Optional matched Qwen report with historical paths and a missing/invalid output | Only test paths included; invalid output remains a failure; missing output makes coverage partial rather than changing the target set. |

Use this hand-calculated metric fixture for I4/reporting tests:

- Five planned/attempted images, true severities `[0,1,2,3,0]`.
- First four predictions have severities `[0,2,2,1]`, with noise/route derived normally; fifth prediction fails.
- Expected evaluated count 4, failed count 1, coverage `4/5`, severity accuracy `1/2`, severity MAE `3/4`, severity macro-F1 `5/12`, noise accuracy `1`, route accuracy `1/2`, severity correct/attempted `2/5`.
- An additional labeled but unattempted image changes planned coverage to `4/6`; it must not become a valid prediction or improve attempted accuracy.

## Notebook and real-data acceptance

1. Validate notebook structure with `nbformat`; save with cleared outputs and no obsolete Qwen runtime imports. Match the project's kernel convention or document the alternate TRM kernel.
2. Execute a reduced CPU smoke configuration end-to-end on temporary synthetic files, writing a split, training checkpoint/history, inference records, metrics, and all CSVs. Verify cells work in order from a fresh kernel and project-root discovery works from the project and notebook directories.
3. Exercise checkpoint-only evaluation in a fresh process, then run the offline report without loading any neural model.
4. Run one forward/backward smoke step on available MPS; on CUDA when that environment is available. Confirm tensors/states are on the requested device. Record untested devices explicitly instead of requiring GCP provisioning for local completion.
5. Validate real dataset counts and run a local full training session using the frozen 140/20/40 split. Evaluate `best.pt` on the 40 held-out images and export the complete report. Every attempted failure must be visible; failures must not be silently repaired. Any unattempted image makes the test pass incomplete.
6. Record measured results, training/inference duration, model size, selected epoch, and recursive budget. Do not impose a target accuracy or claim an improvement over Qwen without matched measurements.

## Implementation completion checklist

- [x] Pinned official source, attribution, and working documented environment.
- [x] Validated person split and native-resolution patch loader.
- [x] Jointly trainable image encoder plus actual shared TRM recurrence.
- [x] Portable best-checkpoint selection based only on validation.
- [x] New supervised notebook using the existing runner and metric schema.
- [x] Existing Qwen regression tests plus new CPU tests pass.
- [x] Synthetic notebook smoke run and checkpoint-only/offline paths verified.
- [x] Real held-out run produced, or any runtime/data blocker clearly documented without inventing results.

Executed checks and any unavailable devices are recorded in [04](04-validation-results.md). The optional matched Qwen comparison is not part of the first implementation.
