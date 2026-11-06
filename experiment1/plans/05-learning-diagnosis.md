# 05 — Tracing the initial model's failure to learn

The selected checkpoint suppresses image-dependent information in the recursive
states. The CNN produces useful differences, but its average pooling is a weak
representation of noise, and the trained recurrence contracts those differences
to almost identical answer vectors. The classifier consequently learns a nearly
constant distribution. This diagnosis applies to our image adaptation and
starting settings; it does not establish a limitation of the official TRM.

The original configuration, checkpoint, and Experiment 1 results are preserved.
Only the saved **140 training and 20 validation images** were traced. Controls
use **eight training images from two people**, one image per severity per person.
No test image pixels were opened and no new held-out results were generated.

## What happens at each stage

### 1. Labels, patch extraction, and input signal

The nine native-scale patches contain a strong severity signal. As a diagnostic,
we compute the standard deviation of a local luminance difference:
`(bottom_right - top_right - bottom_left + top_left) / 2`.
Class centroids fitted on the 140 training images classify **139/140 training**
and **19/20 validation** images correctly; derived routing is correct on all
20 validation images. This is a localization probe, not a replacement TRM result.

| Severity | Training mean residual std | Validation mean residual std |
| --- | ---: | ---: |
| 0 | 0.00708 | 0.01209 |
| 1 | 0.12084 | 0.11486 |
| 2 | 0.16541 | 0.15695 |
| 3 | 0.19854 | 0.19035 |

This argues against label mapping or patch extraction being the source of the
chance-level result. It does not establish that all possible data issues are absent.

### 2. CNN and pooling: noise information exists, but much is discarded

On the trained CNN's last feature map, **76.3%** of the variance of spatial-standard-deviation
features is explained by severity class centroids. For the spatial means actually
retained by average pooling, that fraction is only **0.82%**. This statistic is
descriptive and uses the training split.

A fixed ridge probe, fitted and standardized on training features only, achieves
**80% validation severity accuracy** from the spatial standard deviations,
versus **55%** from the means. The projected patch means still allow a separate
probe to achieve **100% validation route accuracy**. Information therefore
survives the encoder; it is not fully erased there.

The projected vectors have RMS magnitude **0.803**, but their across-image
variation is only **0.0213 RMS**. Most of their energy is common across images.
The official discrete token embeddings are replaced here by continuous CNN
vectors; copying the embedding scale alone does not ensure useful input contrast.

### 3. Recursion: the main observed contraction

Across-image variation of the trained model, measured over all training images:

| Stage | Across-image standard deviation, RMS |
| --- | ---: |
| Mean projected patch vector entering TRM | 0.02125 |
| First working-state update, readout position | 0.00630 |
| Second working-state update, readout position | 0.00131 |
| First answer update | 0.000099 |
| End of first outer refinement, answer readout | 0.000030 |
| End of fourth outer refinement, answer readout | 0.000029 |

The first three reasoner calls are the first H cycle: two working updates, then
an answer update. The next three finish the second H cycle and first outer step.
The answer retains RMS magnitude approximately one throughout, while differences
between images shrink. By the second outer step the probabilities have effectively
converged. More refinements are not improving discrimination in this checkpoint.

This contraction is learned: the seed-matched untrained model's final answer
variation is **0.111**, versus **0.000029** after training. Tiny residual features
can still be amplified by a separately standardized probe, so this is suppression
of usable signal rather than a claim of exact information-theoretic erasure.

### 4. Classifier and gradients

Mean validation probabilities are approximately **[24.19%, 27.06%, 24.25%, 24.51%]**,
with across-image probability variation around one millionth. Every image selects
severity 1, which deterministically maps to route 0.

On the same balanced eight-image training batch, at the first supervised step:

| Gradient L2 norm | Initial model | Selected checkpoint |
| --- | ---: | ---: |
| Encoder | 1.185 | 0.0000438 |
| Reasoner | 8.908 | 0.131 |
| Classifier | 5.301 | 0.273 |

Gradients exist and are finite. However, the encoder gradient has fallen by
roughly **27,000×**, while the classifier can still adjust its shared class
preference. This explains why a nonzero-gradient unit test was insufficient.

Computing gradients through both inner H cycles, with exactly the same forward
values, changes the collapsed encoder norm only from **0.0000438 to 0.0000487**.
Removing that `no_grad` boundary alone does not restore the signal in this
checkpoint. This check does not test removing the outer detached-state boundary
or retraining under full backpropagation.

## Training-only intervention controls

Each control starts from the same seed-matched initialization, uses the same
balanced eight real training images, AdamW learning rate `0.0003`, clipping at 1,
and **400 optimizer updates**. These are small-subset learnability checks, not
generalization estimates or selected experiment configurations. Compute differs
because recursion budgets differ.

| Control | H / L / K | Final training accuracy | Final cross-entropy |
| --- | --- | ---: | ---: |
| Original jointly trained encoder + TRM | 2 / 2 / 4 | 37.5% | 2.07859 |
| Only reduce H | 1 / 2 / 4 | 100% | 0.00185 |
| Only reduce L | 2 / 1 / 4 | 100% | 0.08773 |
| Only reduce K | 2 / 2 / 1 | 100% | 0.00077 |
| Reduce H, L, and K | 1 / 1 / 1 | 100% | 0.00057 |
| Frozen encoder, raw vectors + TRM | 2 / 2 / 4 | 25% | 1.38617 |
| Same frozen encoder, standardized vectors + TRM | 2 / 2 / 4 | 100% | 0.00098 |
| CNN means + direct classifier | — | 100% | 0.00412 |
| CNN means and std + direct classifier | — | 100% | 0.03928 |

The original model's loss and discrimination fluctuate rather than improving
reliably on this subset. Each individual recursion reduction is sufficient for
subset memorization at this seed; the results do not identify a unique best budget.

The frozen-token comparison isolates scaling while keeping the encoder frozen
in both cases. Standardization subtracts a per-feature mean and divides by a
per-feature standard deviation fitted on those eight images and nine patches;
the learned readout token is unchanged. It does not use severity labels.

The CNN-only moment control begins with exactly the same function as the mean
control: the additional standard-deviation projection weights start at zero.
Both CNN-only controls learn, confirming that the conversion component can be
trained on real image data independently of recursion.

## Interpretation and next experiment

The strongest supported explanation is a combination of **weak input contrast**
and **unstable/contracting recurrent dynamics**. The average-pooled image adapter
makes noise cues unnecessarily subtle; the trained recurrence largely suppresses
those differences, and classifier updates can reduce loss toward the class prior
without learning useful image discrimination.

A minimal next experiment should preserve spatial variability in the thin
converter, standardize features using training statistics, and begin with a
reduced recursion budget. First require reliable real-subset overfitting, then
train on the full training split and select using validation. Increase recursion
one setting at a time after the simpler configuration learns. The ordinary
pipeline and original severity-to-route mapping can remain unchanged.

We have not isolated the exact first update that causes collapse in the original
full run, learning-rate sensitivity, seed sensitivity, or full-data generalization
of these interventions. Only the original best checkpoint was saved, so a replay
with intermediate traces would be needed to establish the collapse timeline.
The eight-image successes may reflect memorization, and validation contains only
five people. No production defaults have been changed based on these diagnostics.

## Artifacts and reproduction

- [Stage statistics and frozen ridge probes](../outputs/trm_diagnostics/20261024_stage_trace_final/stages.csv).
- [Per-step probabilities](../outputs/trm_diagnostics/20261024_stage_trace_final/probabilities.csv).
- [Gradient traces](../outputs/trm_diagnostics/20261024_stage_trace_final/gradients.json) and [truncation check](../outputs/trm_diagnostics/20261024_stage_trace_final/truncation_check.json).
- [Centroid probes](../outputs/trm_diagnostics/20261024_stage_trace_final/centroid_probes.json), [trace manifest](../outputs/trm_diagnostics/20261024_stage_trace_final/manifest.json), and frozen feature tensors beside them.
- [Main controls](../outputs/trm_diagnostics/20261024_learning_controls/history.csv), [individual recursion controls](../outputs/trm_diagnostics/20261024_recursion_controls/history.csv), and [matched scaling controls](../outputs/trm_diagnostics/20261024_scaling_controls/history.csv).
- [Combined control table](../outputs/trm_diagnostics/controls_summary.csv) and [machine-readable summary](../outputs/trm_diagnostics/diagnostic_summary.json).
- [Unit-test log](../outputs/trm_diagnostics/unit_tests.txt): **67 tests, 64 passed, 3 skipped** (CUDA, sandbox-hidden MPS, opt-in overfit diagnostic). Seven new tests check trace fidelity, no parameter mutation, gradient measurement, probe behavior, and the moment control's initialization.

Use new output directories:

```sh
OMP_NUM_THREADS=2 .venv/bin/python scripts/diagnose_trm.py --output outputs/trm_diagnostics/new_trace
OMP_NUM_THREADS=2 .venv/bin/python scripts/diagnose_trm_controls.py --output outputs/trm_diagnostics/new_controls
OMP_NUM_THREADS=2 .venv/bin/python scripts/diagnose_trm_controls.py --output outputs/trm_diagnostics/new_recursion --variants h1 l1 k1
OMP_NUM_THREADS=2 .venv/bin/python scripts/diagnose_trm_controls.py --output outputs/trm_diagnostics/new_scaling --variants raw_frozen_tokens normalized_frozen_tokens
```
