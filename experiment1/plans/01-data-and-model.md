# 01 — Dataset and model specification

Depends on: [plan overview](README.md). Followed by: [training and notebook](02-training-and-notebook.md).

## Dataset and split

Use `build_dataset(PROJECT_PATH / "data/files")` without changing its label mapping:

| Folder | Severity | Noise | Route |
| --- | --- | --- | --- |
| `images/` | `0` | `0` | `0` |
| `augmentation/1/` | `1` | `1` | `0` |
| `augmentation/2/` | `2` | `1` | `1` |
| `augmentation/3/` | `3` | `1` | `1` |

Validate 200 unique image paths, 50 person IDs, and exactly one row per person/severity. Missing files, incomplete groups, duplicate rows, or invalid labels must fail before training. The dataset root and person IDs select files and splits; they must not become model features.

In `src/trm_data.py`, implement `create_subject_split(dataset, seed=42)`:

1. Sort the unique string `subject_id` values lexicographically, then permute using a seeded NumPy generator.
2. Assign the first 35 IDs to train, next five to validation, remaining ten to test.
3. Retain all four versions of each person in their assigned partition; preserve source frame ordering within each partition.
4. Save explicit ID lists and project-relative image paths, labels, and SHA-256 file hashes in `split.json`. The saved manifest is authoritative for checkpoint reload and comparison.
5. Resolve relative paths against the current project root so a copied dataset works on GCP. Reject missing, overlapping, or changed manifest entries on reload; do not silently regenerate a different split.

Expected class counts are 35/5/10 per severity in train/validation/test. A separate smaller fixture mode is allowed for tests, with explicit group counts. Never apply the Qwen few-shot reference holdout to this supervised split.

## Image-to-vector component

Implement an image loader and a PyTorch dataset in `src/trm_data.py`. Input samples contain patch tensors and an integer severity target; file paths and person IDs remain metadata.

Default preprocessing, identical for training, validation, and inference:

- Open with Pillow in a context manager; convert to RGB.
- Keep `exif_transpose=False`, following the original notebook, and record it. Do not use EXIF or JPEG metadata as model features.
- Do not resize or denoise the whole image. Sample nine 128×128 patches on a row-major 3×3 grid.
- For each axis, centers are at fractions `0.25`, `0.50`, `0.75` of `(dimension - 1)`. Set the crop origin to `floor(center - patch_size / 2)`, clamped to `[0, dimension - patch_size]`. Crop exactly `patch_size` pixels.
- Reject dimensions smaller than `patch_size` with a clear error. All real dataset images must pass this check before training. Smaller configurable patches are allowed in synthetic tests.
- Convert to float32 channel-first tensors and divide RGB values by 255. No dataset-fitted normalization, random crops, or additional augmentation in version 1.

Per image the resulting shape is `[9, 3, 128, 128]`; a batch is `[B, 9, 3, 128, 128]`. This native-scale patch representation is an explicit difference from Qwen's default whole-image resizing.

In `src/trm_model.py`, implement `PatchEncoder`: flatten batch and patch axes, apply shared `Conv2d` layers `3→16→32→64`, each with kernel 3, stride 2, padding 1 and GELU; adaptive-average-pool to 1×1; project 64 features to hidden width 128. Reshape to `[B, 9, 128]`. Use no batch normalization, dropout, pretrained weights, or encoder freezing in version 1.

## TRM adaptation

Vendor the required official modules, retaining upstream attribution. Rewrite package imports only as needed to avoid a global `models` namespace; list those changes in provenance. Reuse the official reasoning block/module classes from that pinned source rather than replacing them with a generic recurrent layer.

Implement `VisionTRM` in `src/trm_model.py`:

- Prepend a learned, shared readout input vector to the nine patch vectors, giving `x: [B, 10, D]` (the readout is position zero).
- Use the official input scale `sqrt(D)` and rotary positions. Use two shared attention/MLP reasoning layers, `D=128`, four heads, and expansion 4. Head dimension must be even and width divisible by head count.
- Initialize answer and working states using the upstream initialization method; reset them for every new image batch.
- Adapt the recurrence to continuous `x`. Do not quantize pixels or visual vectors. Omit integer-token embeddings, person/puzzle embeddings, token-language output, and adaptive-halting heads from the instantiated local model.
- Apply a learned linear `D→4` classifier to answer position zero. Return severity logits `[B, 4]`.

The preserved computation is:

```text
for each outer refinement step:
    for h in 1..H_cycles:
        repeat L_cycles times:
            working = shared_reasoner(hidden_states=working, input_injection=answer + x)
        answer = shared_reasoner(hidden_states=answer, input_injection=working)
    severity_logits = classifier(answer[:, 0])
    detach answer and working before the next outer step
```

Follow upstream gradient truncation: the first `H_cycles - 1` cycles run without gradients; the last cycle supports gradients. The encoder's `x` must remain attached for the final cycle. Defaults are `H_cycles=2`, `L_cycles=2`, `refinement_steps=4`. Use all four outer steps during both training and evaluation; no learned early stopping or cross-image state reuse.

Expose separate operations to encode patches, initialize state, and execute one refinement step so the trainer can supervise each step. An inference convenience method executes the configured number of steps and returns only the final logits. Keep tensor creation device-aware, including initial states and rotary buffers.

This is an image-classification TRM adaptation with fixed recursion and a new readout, not the unchanged official ACT training recipe. Document these differences alongside the pinned upstream reference.

## Component interfaces

```python
create_subject_split(dataset, seed=42) -> split_manifest
load_subject_split(path, dataset, project_root) -> split_manifest
load_image_patches(image_path, preprocess_config) -> Tensor
FaceSeverityDataset(frame, preprocess_config)  # patches + severity target
PatchEncoder(config).forward(patches) -> Tensor  # [B, P, D]
VisionTRM(config).encode(patches) -> Tensor
VisionTRM(config).initial_state(batch_size, device) -> state
VisionTRM(config).refine_once(encoded, state) -> (detached_state, logits)
VisionTRM(config).forward(patches) -> Tensor  # final [B, 4] logits
```

Allow reduced width, patch count, patch size, and cycles in test configurations. Default configuration validation must catch zero/negative counts and incompatible dimensions before model execution. Report actual encoder, reasoning, classifier, and total trainable parameter counts; do not label this customized model as the paper's exact 7M model.
