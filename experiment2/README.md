# Experiment 2 — Kaggle GPU runs

A self-contained workspace for face-image degradation classification and
denoising routing. It needs no files from `experiment1`. Source, configs,
notebooks, tests, and the MIT-licensed TRM blocks belong in Git; images, model
caches, checkpoints, and results are regenerated and gitignored.

| Workflow | Precision | Evaluation |
| --- | --- | --- |
| Qwen single shot (zero shot, no examples) | FP16 weights + efficient SDPA | 196 images, reference subject `0` excluded |
| Qwen few shot | FP16 weights + efficient SDPA | Same 196 targets; four labeled images of subject `0` |
| Supervised visual TRM | FP16 AMP + gradient scaling | 35/5/10 disjoint subjects: 140 train / 20 validation / 40 test images |
| Noise-aware ViTRM | FP16 AMP + gradient scaling | Same subject split and severity/routing labels as TRM |

Both Qwen modes use `Qwen/Qwen3.5-0.8B` at revision
`2fc06364715b967f1860aea9cf38778875588b17`. “Single shot” here means the
existing single-image, zero-shot workflow, not one labeled demonstration.
The TRM is trained from scratch using the existing visual adapter around the
vendored upstream recursion blocks; it is not a pretrained vision model.

## Run in Kaggle

1. Commit and push `experiment2` to GitHub. The notebooks clone the published
   source; local uncommitted files are unavailable to Kaggle.
2. Import one notebook into Kaggle and enable **Internet** and a **GPU**
   accelerator. A single T4 is sufficient for the previously tested Qwen workload.
3. Run the cells in order. The first cell sparsely checks out only `experiment2`
   and installs its Python dependencies, retaining Kaggle's CUDA-enabled PyTorch.
   Change `REF` to a published commit, tag, or branch if needed. Each run also
   saves a complete source/config snapshot and hashes.
4. Use `SELECTION = "smoke"` first on a new runtime, then `"full"`. Each workload
   runs in a separate Python process; exiting releases its GPU allocations.
5. Download the generated archive or save a Kaggle notebook version with outputs
   before shutting down. Runtime storage and caches are disposable.

Notebooks:

- [Qwen single shot](notebooks/qwen_single_shot.ipynb)
- [Qwen few shot](notebooks/qwen_few_shot.ipynb)
- [Supervised TRM](notebooks/trm_supervised.ipynb)
- [Supervised ViTRM](notebooks/vitrm_supervised.ipynb)

The notebooks default to full runs. The Qwen smoke selection is fixed in advance:
subjects `27`, `14`, and `9`, all four severities (12 targets). The `largest`
selection runs `augmentation/3/18.jpg` with the same references and image
settings. The TRM smoke selection trains for one epoch on the same disjoint
partitions and evaluates its selected checkpoint; it tests execution, not model
quality. Smoke and full results have separate output directories.

No proxy URL, API token, or service-account file is needed in source. For a
private repository, authenticate Git through your runtime's secret mechanism;
do not embed credentials in the notebook URL.

## Command line

From a Kaggle terminal, with the repository already checked out:

```bash
cd /kaggle/working/CS297-research-project-em/experiment2
python -m pip install -r requirements.txt
python -m src.run qwen_single_shot --selection smoke
python -m src.run qwen_few_shot --selection largest
python -m src.run qwen_few_shot --selection full
python -m src.run trm_supervised --selection full
python -m src.run vitrm_supervised --selection smoke
python -m src.run vitrm_supervised --selection full
```

Use `--gpu 1` to select the second T4. Each process uses one GPU, with no CPU
offload or cross-GPU model sharding. Independent runs may use separate GPUs;
each GPU keeps its own model copy. Default operation is sequential. Use
`--output /kaggle/working/my_fresh_run` to choose a result location; an existing
directory is rejected. Interrupted runs retain completed image batches and
training history/checkpoints. Restarting creates a new run; training and
inference resume are not implemented.

## ViTRM evaluation

This independent adaptation follows the separate image/memory/answer states and
shared Transformer updates in [ViTRM, equations 2–3](https://arxiv.org/html/2603.19503v2#S3.SS3).
It is trained from scratch; it is not the authors' implementation or a reproduction
of their CIFAR results. The original TRM configuration and vendored blocks remain
the baseline. Both workflows predict severity and denoising routing; neither
executes a denoiser.

`configs/vitrm.json` keeps the nine native-scale 128×128 crops, width 128, and
35/5/10 person split. Each crop produces a 4×4 grid of tokens containing local
CNN feature means/standard deviations and RGB means/standard deviations, projected
to width 128. RGB moments provide a path around CNN downsampling. Moment reductions
use FP32, with no per-image variance normalization. Learned crop/cell embeddings
identify all 144 image tokens. Four learned memory tokens and one learned answer
token are reset for every batch. A single shared two-layer, pre-norm Transformer
stack alternates between:

1. Process `[image, answer, memory]`, retaining only updated memory tokens.
2. Process `[answer, memory]`, retaining only the updated answer token.
3. Read four severity logits from the normalized answer.

Image tokens stay fixed within an unroll. The default is one memory update and
one refinement. Full attention still scales quadratically with the concatenated
sequence length. There is no halting head, extra pretrained backbone, or resize
of the whole image. Training uses the mean cross-entropy across all refinements,
full unrolled gradients, and **one optimizer update per batch**. This intentionally
differs from the paper's detached, per-supervision-step optimizer updates and
the existing TRM trainer. FP16 training uses gradient scaling and clipping.

Use validation-only training for architecture/depth selection. These commands
create separate runs and never infer test pixels:

```bash
python -m src.run vitrm_supervised --refinement-steps 1 --validation-only
python -m src.run vitrm_supervised --refinement-steps 2 --validation-only
python -m src.run vitrm_supervised --refinement-steps 4 --validation-only
```

After choosing the protocol, evaluate its saved checkpoint without retraining:

```bash
python -m src.run vitrm_supervised --checkpoint outputs/YOUR_RUN/checkpoints/best.pt
```

`--config path/to/config.json` accepts a complete training configuration for
seed, depth, or model ablations. A checkpoint fixes its architecture and depth;
checkpoint-only evaluation rejects config/depth overrides. The original split
membership and image hashes are verified before evaluation. Checkpoints are
selected by final-refinement validation loss, using no test predictions.
Training and evaluation output directories must be fresh. CPU FP32 execution
is available for local workflow checks:

```bash
python -m src.run vitrm_supervised --device cpu --selection smoke --validation-only
python -m src.run vitrm_supervised --device cpu --checkpoint outputs/YOUR_RUN/checkpoints/best.pt
```

Alongside the existing JSON/CSV reports, ViTRM writes
`refinement_predictions.csv` (probabilities, predictions, and model-only elapsed
time for the **entire unroll**, excluding image loading/transfers) and
`refinement_summary.csv` (per-step severity accuracy/macro-F1/MAE, routing errors,
and planned coverage) inside `experiment_002/run_001/`. The model timer includes
all refinements and is repeated on each step's row; it is not per-step latency
or a warmed-up throughput benchmark. Final-step metrics agree with the ordinary
saved-run report. Training history adds the mean pre-clipping encoder gradient
norm and per-refinement validation loss/accuracy/between-image probability
standard deviation to help detect collapse. The training manifest records the
ViTRM implementation identifier and paper URL; it does not attribute the new
model to the vendored TRM commit.

Use multiple initialization seeds and compare a direct classifier with the
same moment encoder before attributing gains to recursion. Crops remain grouped
by source image and person; 144 tokens do not constitute 144 independent samples.
Smoke and synthetic checks establish execution/learnability, not real-data
accuracy. CUDA validation still requires a GPU runtime.

### Controlled moment-encoder comparison

For the focused encoder-versus-recursion comparison, use
`src.encoder_comparison`. It implements three explicitly named variants:

| Variant | Visual input | Recursive reasoning |
| --- | --- | --- |
| `linear_vitrm_1/2/4` | Flattened 32×32 RGB cells projected by one linear layer | Shared ViTRM at 1, 2, or 4 refinements |
| `moment_vitrm_1/2/4` | Existing CNN/RGB mean-and-standard-deviation moments | Shared ViTRM at 1, 2, or 4 refinements |
| `moment_direct` | Existing CNN/RGB moment encoder | Mean pooling plus a direct classifier |

All variants use the same 144-token layout, positional embeddings, subject
split, initialization seeds, minibatch order, AdamW settings, epoch budget,
and validation-loss checkpoint selection. The test pixels are loaded only
after every variant and seed has selected a checkpoint. This isolates the
effect of the visual encoder from the effect of recursive reasoning more
cleanly than comparing unrelated model families. The linear encoder is a
cell-level linear projection, matching the paper's minimal visual-front-end
idea while retaining the experiment's 144-token spatial layout.

Run the comparison on Kaggle with a validated dataset root:

```bash
python -m src.encoder_comparison \
  --dataset-root /kaggle/input/your-dataset \
  --output outputs/encoder_comparison \
  --device cuda \
  --epochs 10 \
  --seeds 42 43 44 \
  --refinement-steps 1 2 4
```

To reuse an existing subject split, pass its saved manifest:

```bash
python -m src.encoder_comparison \
  --dataset-root /kaggle/input/your-dataset \
  --reference-split outputs/YOUR_REFERENCE_RUN/split.json \
  --output outputs/encoder_comparison \
  --device cuda \
  --epochs 100 \
  --seeds 42 43 44 \
  --refinement-steps 1 2 4
```

The runner writes `comparison_runs.csv`, `comparison_summary.csv`, one
checkpoint and training history per variant/seed, and `protocol.json`.
`comparison_summary.csv` reports mean and standard deviation across
initialization seeds; those repetitions are not independent datasets. The
direct classifier is trained once per seed, while each listed ViTRM depth is
trained as a separate model. Each checkpoint is selected on validation loss;
test pixels are loaded only after all depth/encoder checkpoints are selected.
`comparison_runs.csv` also records `test_step_N_*` metrics for each available
refinement step, allowing within-checkpoint prefix comparisons for the deeper
ViTRM models.

`src.moment_ablation` compares mean-pooled linear and MLP classifiers, the old
TRM recurrence, and separately trained one- and two-refinement ViTRMs. Defaults
are 10 epochs and initialization seeds 42, 43, and 44, with the same 140/20/40
subject split. Every variant receives identical initial moment-encoder and
position weights within a seed, the same minibatch order, AdamW settings, and
180 optimizer updates. Encoder weights then train independently end to end;
they are not frozen features. Model sizes and computation costs differ.

The old TRM uses all 144 image tokens and retains H=2, L=2, four outer steps,
additive injection, sqrt(width) scaling, RoPE, fixed initial states, and its
original gradient boundaries. For equal optimizer-update and image-exposure
budgets, the comparison averages its four supervision losses before one optimizer
step per batch. This differs from the historical per-refinement optimizer
updates and is recorded explicitly in `protocol.json`.

```bash
python -m src.moment_ablation \
  --dataset-root /path/to/files \
  --reference-split outputs/YOUR_REFERENCE_RUN/split.json \
  --output outputs/my_fresh_moment_comparison
```

The runner verifies dataset hashes and subject membership against the reference,
caches only training/validation pixels initially, and fixes all 15 checkpoints
using validation loss before loading test pixels. It saves individual checkpoints,
training histories, standard evaluation reports and per-refinement reports,
`comparison_runs.csv`, `comparison_summary.csv`, and paired seed differences.
Reported standard deviations describe variation across model initialization seeds
on one split. Reusing the previously inspected 40-image test cohort makes this
an exploratory comparison, not a new independent confirmation set.

### Does carrying state through more refinements help?

`src.recursion_ablation` trains a **single four-refinement ViTRM**, supervising
steps 1, 2, 3 and 4 equally with mean cross-entropy and full unrolled gradients.
It selects one checkpoint by step-4 validation cross-entropy. Evaluation encodes
each image once and reads predictions at steps **1, 2 and 4 from one unroll**;
weights, image tokens and initial states are identical across those prefixes.
The model API rejects depths beyond the configured training depth.

The comparison includes a separately trained `shared_1` reference and an
`untied_4` ordinary-depth control. The latter has the exact same operations,
sequence lengths, state flow, shared encoder and output head, but **independent
Transformer stacks at every memory and answer call**. With one memory update
and two Transformer layers per stack, four stages perform eight stack calls
(16 layer applications). All eight untied stacks initially copy the shared
model's weights, so both deep models start with identical functions. Their
weights subsequently learn independently. This is a feed-forward unrolling
with ordinary untied depth, rather than a conventional CLS-token ViT. It
controls computation structure, not parameter count: 442,020 parameters for
either shared model versus 3,217,828 for the untied control. Prefix improvements
within a checkpoint measure the benefit of additional computation/state updates;
the untied comparison tests whether weight sharing is useful relative to
ordinary depth. It does not isolate state carry from all other computation.

```bash
python -m src.recursion_ablation \
  --dataset-root /path/to/files \
  --reference-split outputs/YOUR_REFERENCE_RUN/split.json \
  --output outputs/my_fresh_recursion_comparison \
  --epochs 10 --seeds 42 43 44 --validation-only

# Evaluate all selected checkpoints without retraining:
python -m src.recursion_ablation \
  --output outputs/my_fresh_recursion_comparison --evaluate-only
```

Omit `--validation-only` to train all predefined models before evaluating them
in the same invocation. All variants receive the same initial encoder/positions,
minibatch order within seed, learning rate, epochs and optimizer-update budget.
The runner uses CPU FP32 and two threads by default. Ten epochs are a quick
screening budget, not evidence that each model has converged. It verifies the
reference subject split, labels and file hashes, and loads test pixels only
after all checkpoints are selected. Reusing the old test cohort remains
exploratory. Initialization seeds are repeated fits on one split, not additional
independent test images.

Each model directory and the combined output contain:

- `prefix_predictions.csv`: per-image probabilities, cross-entropy, predicted
  severity, absolute severity error and routing error at steps 1, 2 and 4.
- `prefix_metrics.csv`: accuracy, cross-entropy, severity MAE, routing error
  counts, false positives and false negatives at each requested prefix.
- `image_transitions.csv`: paired per-image corrections, regressions and metric
  changes for 1→2, 1→4 and 2→4, using the same checkpoint.
- `transition_metrics.csv`: corrections/regressions, net corrections, routing
  corrections/regressions and changes in cross-entropy, MAE and routing errors.
  A negative loss/error delta means improvement. A correction means incorrect
  severity becomes correct; an incorrect answer changing to another incorrect
  answer is reflected in loss/MAE rather than counted as a correction.
- `prefix_timing.csv`: median CPU time from identical encoded tokens, with two
  warmups and five repeats on one batch. This excludes the encoder and loading;
  it is a local diagnostic rather than a deployment throughput measurement.

`comparison_summary.csv` aggregates metrics across initialization seeds.
`selected_checkpoints.json`, training histories, protocol, source snapshot,
environment and split manifests preserve how the result was obtained. The
checkpoints use a dedicated comparison schema and are loaded by this runner.
Routing continues to mean predicted severity ≥2; no denoiser is executed.

### Five-fold subject-grouped evaluation

`src.recursion_crossval` repeats the same three model variants and initialization
seeds over five predefined outer test folds. It takes a previous
`recursion_ablation` run as its protocol reference, preserving model settings,
training budget, preprocessing and dataset hashes. With 50 subjects, each fold
has 35 training, 5 validation and 10 test subjects (140/20/40 images). The seeded
subject permutation rotates by ten people per fold; fold one exactly preserves
the reference split when its counts are 35/5/10. All four images of a person
stay together, and every subject is tested in exactly one fold. Validation
subjects are selected from the remaining 40 people, never from that fold's test
group. A separate model is trained for every fold/variant/seed.

```bash
python -m src.recursion_crossval \
  --reference-run outputs/YOUR_PREVIOUS_RECURSION_RUN \
  --output outputs/my_fresh_five_fold_comparison

# Continue after an interruption, keeping the saved protocol:
python -m src.recursion_crossval \
  --output outputs/my_fresh_five_fold_comparison --resume
```

Defaults inherit the reference run's 10 epochs, seeds 42/43/44 and CPU FP32
execution: **45 fits**, each with 180 optimizer updates. A new run can override
`--epochs`, `--seeds` or `--threads`; resume rejects protocol overrides. A moved
but byte-identical dataset can be supplied with `--dataset-root` on a new run.
Completed fits are reused only after checking their protocol, update count and
selected-checkpoint hash. An interrupted fit restarts from scratch, retaining
its incomplete files in a separate directory. Resume also verifies the source
snapshot, preventing mixed implementations within one comparison.

Every checkpoint is selected by its own validation loss before **any** outer
test prediction is generated. Deterministic image crops are cached in RAM as
needed. An image can train a model in one fold and test a different model in
another; only each fit's own training partition reaches its optimizer. There
is no model-state reuse across folds. All model prefixes continue to use the
same checkpoint and fixed image tokens within each unroll.

Each `fold_01/` through `fold_05/` retains the full split, checkpoints, histories,
per-image prefix predictions and correction/regression reports. Combined files:

- `oof_predictions.csv` and `oof_image_transitions.csv`: every image's held-out
  predictions and paired changes, with fold, seed and checkpoint identifiers.
- `oof_prefix_metrics.csv` and `oof_transition_metrics.csv`: metrics pooled over
  all **200 unique held-out images per seed**, rather than treating five folds
  as independent repetitions.
- `comparison_summary.csv` and `transition_summary.csv`: mean performance and
  corrections/regressions across the three initialization seeds. Accuracy
  standard deviation describes initialization variation on these fixed folds.
- `fold_prefix_metrics.csv`, `fold_transition_metrics.csv` and
  `fold_prefix_timing.csv`: individual-fold results for diagnosis.
- `coverage.json`: checks that each image has exactly one held-out prediction
  per model/seed/prefix; missing or duplicated predictions fail aggregation.

The five training sets overlap, and the dataset has already informed prior
experiments. This is broader exploratory evaluation, not a new independent
confirmation dataset. It adds held-out coverage, not additional source images.

## Data

`kagglehub.dataset_download` obtains
[`trainingdatapro/human-faces-augmentation-image-restoration`](https://www.kaggle.com/datasets/trainingdatapro/human-faces-augmentation-image-restoration)
directly from Kaggle, reusing the KaggleHub cache or Kaggle-managed input mount.
A validated previous download is reused through `data/dataset_source.json`.
To use an already attached dataset explicitly, set notebook `DATASET_ROOT` or
pass `--dataset-root /kaggle/input/...`. It accepts a wrapper directory containing
the dataset. An invalid explicit root fails rather than silently replacing data.
No images are copied into this repository.

All 200 images are checked for decoding, minimum dimensions, complete
subject/severity membership, and SHA-256 content. Each run saves the dataset
digest, per-image hashes and dimensions, source handle, and resolved version
when KaggleHub exposes it. A future dataset revision can change the digest;
preserve the run manifest for comparison. Expected layout:

```text
images/<subject>.jpg             clean (severity 0)
augmentation/1/<subject>.jpg     weak
augmentation/2/<subject>.jpg     medium
augmentation/3/<subject>.jpg     strong
```

## Precision and evaluation choices

Qwen retains the original prompt, full RGB images, no EXIF transpose, no custom
pixel cap or crops, processor size settings `shortest_edge=65536` and
`longest_edge=16777216`, greedy decoding, and 128 maximum new tokens. Batch size
is one. Few shot uses four labeled user/assistant turns before the target.

Efficient attention means PyTorch's `EFFICIENT_ATTENTION` SDPA backend. Native
grouped-query dispatch is disabled through a non-recording callable so that
Transformers explicitly expands KV heads on T4. A small causal/noncausal CUDA
correctness check runs before model loading. Unsupported backends fail visibly;
there is no silent math-attention fallback. Optional FlashAttention, causal-conv,
and flash-linear-attention packages are not required. This follows the corrected
benchmark, including its fix for the recording mock that retained GPU tensors.

TRM uses FP16 `torch.autocast` for training, validation, and test inference,
with `torch.amp.GradScaler` for training. Master parameters, AdamW state, and
numerically sensitive reductions remain FP32. Gradient clipping follows
unscaling; overflow skips are recorded. The recursion budget, nine native-scale
128×128 patches, model widths, split seed 42, validation-loss selection,
100-epoch limit, and patience 15 match the previous supervised setup. Only
train and validation pixels feed learning/selection; the 40 test images are
inferred after checkpoint selection.

Every output is checked against the existing strict JSON schema. Markdown
fences remain failures, even when the enclosed labels would be correct. This
matters because the largest-image benchmark returned fenced JSON in both
precisions. Reports expose valid-only accuracy alongside **Correct / planned**
and failure coverage, so malformed responses do not disappear from the result.
Qwen's 196-image evaluation and TRM's 40-image held-out evaluation are different
cohorts; their aggregate scores are not a matched comparison.

The earlier corrected few-shot screening on T4 measured about 3.37× throughput
over FP32 on the fixed 12-image subset with identical labels. The largest input
used about 7.0 GiB peak allocated memory in FP16. These are prior screening
results, not a completed Experiment 2 run or an accuracy guarantee. The new
TRM AMP training path still needs verification on a live Kaggle runtime.

## Files and results

```text
configs/         Qwen revision, original prompt, supervised TRM configuration
notebooks/       four Kaggle entry points
src/             data validation, model adapters, training, inference, reports
tests/           local contract/data/model checks and opt-in CUDA coverage
vendor/          minimal existing TRM source, provenance, and MIT license
requirements.txt Kaggle add-on dependencies; uses the installed CUDA PyTorch
```

Each `outputs/<task>_<selection>_<timestamp>/` contains `run.json`, dataset and
source manifests, and `experiment_002/run_001/` with raw batch JSON, labels,
metrics, predictions/failures CSVs, coverage, and evaluation summary. Qwen also
saves per-image GPU allocation/timing telemetry and the selected target list.
TRM adds split/config files, training history, training manifest, and the
validation-selected `checkpoints/best.pt`. Checkpoints contain weights/config
and split identity; optimizer/scaler resume state is not saved.

The validated Qwen runtime was Python 3.13, torch 2.11.0+cu130,
torchvision 0.26.0+cu130, transformers 5.20.2, and accelerate 1.15.0. Dependencies
that control Qwen attention are pinned; supporting-library versions and GPU
identity are captured for every run. A CUDA-capable PyTorch install is required
outside Kaggle. Avoid installing Experiment 1's requirements over this runtime.

Run checks from this directory after installing dependencies and PyTorch:

```bash
python -m unittest discover -s tests -v
```

CUDA-only checks skip when no GPU is available. No test downloads the real
dataset or model. The AMP implementation follows
[PyTorch's mixed precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples),
and download/cache behavior follows the
[official KaggleHub documentation](https://github.com/Kaggle/kagglehub).
