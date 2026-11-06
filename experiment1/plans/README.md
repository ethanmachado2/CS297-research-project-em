# Experiment 1 with TRM — implementation plan

Status: implemented. See [validation results](04-validation-results.md) for executed checks and the initial held-out experiment. The documents below retain the agreed implementation specification.

The initial model did not learn useful image discrimination. See the subsequent
[stage-by-stage learning diagnosis](05-learning-diagnosis.md) for measured traces,
training-only intervention controls, and the proposed next experiment.

## Objective

Train a lightweight image encoder and a Tiny Recursive Model (TRM) to predict the existing face-image augmentation severity. Derive noise and denoising-route labels from severity, then evaluate through the existing Experiment 1 runner.

```text
image → nine native-resolution RGB patches → small shared CNN
      → image vectors → recursive TRM → severity 0–3 → existing JSON output
```

This is a supervised counterpart to the Qwen experiment, with the same dataset, label meanings, prediction schema, and metrics. It is not a zero-shot experiment or an exact reproduction of the upstream puzzle benchmarks. Multimodal orchestration, policy learning, authentication, denoising execution, and additional model baselines are outside this first slice.

## Read and implement in this order

1. [Data, image encoder, and TRM](01-data-and-model.md).
2. [Training, notebook, and existing pipeline integration](02-training-and-notebook.md).
3. [Test cases and completion criteria](03-tests-and-acceptance.md).

## Conventions to retain

- Treat “workbook” as a Jupyter notebook, matching the two existing `.ipynb` experiments. Add `notebooks/cs297-experiment1-em-local-trm.ipynb`.
- Reuse `src/image_quality_pipeline.py` for dataset construction, JSON validation, checkpoints, and inference progress; reuse `src/evaluation.py` for metrics.
- Follow the notebook's project-root lookup, named configuration cells, UTC timestamp plus UUID session IDs, restart-and-Run-All workflow, per-image evaluation, and offline report/export section.
- Keep reusable training/model logic under `src/`; notebook cells coordinate those components.
- Use `unittest`, temporary directories, and small synthetic fixtures as in the existing tests.
- Preserve existing Qwen notebooks, output folders, labels, metric definitions, and default behavior.

## Decisions and starting defaults

| Item | Decision |
| --- | --- |
| Training permission | User explicitly accepted training the TRM for this experiment. |
| Identity grouping | User confirmed that each of the 50 filename IDs is a different person. |
| Dataset split | 35 training / 5 validation / 10 test people: 140 / 20 / 40 images. |
| Image representation | Nine fixed 128×128 patches; shared small CNN, trained jointly with TRM. |
| TRM | Two shared reasoning layers; width 128; four attention heads. |
| Recursion | `H_cycles=2`, `L_cycles=2`, four outer refinement steps. |
| Prediction | Four severity logits; derive noise and route in Python. |
| Runtime | Local MPS when available, otherwise CPU; optional CUDA configuration later. |
| Precision | Float32 initially on all devices. |
| Reporting | One held-out test pass using the best validation checkpoint. |

These are runnable starting settings, not claimed optimal settings. Configuration changes must be saved with the run. No further user decision is required to implement this version. Confirmed identity grouping resolves the only clarification raised during specification.

## Implemented deliverables

```text
configs/trm_experiment1.json
src/trm_data.py
src/trm_model.py
src/trm_training.py
src/trm_adapter.py
src/trm_reporting.py                     # model-free CSV exports
vendor/tiny_recursive_models/             # required official modules + license/provenance
notebooks/cs297-experiment1-em-local-trm.ipynb
tests/test_trm_data.py
tests/test_trm_model.py
tests/test_trm_training.py
tests/test_trm_integration.py
tests/test_trm_notebook.py
scripts/validate_trm_notebook.py
requirements-trm.txt
```

Also make the small shared-runner extension specified in document 02 and add the new workflow to `PIPELINE.md`. Do not create a separate evaluation framework.

## Upstream foundation

Built on the [official SamsungSAILMontreal/TinyRecursiveModels repository](https://github.com/samsungsailmontreal/tinyrecursivemodels) at commit `c01103738605ba39d1430519b1ee0c62f4c707f8`. Source-file hashes are recorded in [PROVENANCE.json](../vendor/tiny_recursive_models/PROVENANCE.json), the MIT license is preserved, and only the required import closure is vendored. Notebook execution does not fetch a floating branch.

The official [reasoning implementation](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/models/recursive_reasoning/trm.py) supplies the shared reasoning blocks and recurrence. The proposed image input, classifier, fixed-step training wrapper, and notebook are local adaptations. Upstream [training code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/pretrain.py) uses CUDA explicitly; use a small local device-aware trainer instead.
