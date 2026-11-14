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
```

Use `--gpu 1` to select the second T4. Each process uses one GPU, with no CPU
offload or cross-GPU model sharding. Independent runs may use separate GPUs;
each GPU keeps its own model copy. Default operation is sequential. Use
`--output /kaggle/working/my_fresh_run` to choose a result location; an existing
directory is rejected. Interrupted runs retain completed image batches and
training history/checkpoints. Restarting creates a new run; training and
inference resume are not implemented.

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
notebooks/       three Kaggle entry points
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
