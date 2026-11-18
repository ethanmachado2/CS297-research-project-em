"""Person-disjoint data manifests and deterministic native-scale image patches."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from .image_quality_pipeline import validate_prediction

PARTITIONS = ("train", "validation", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_preprocessing(config):
    if set(config) != {"patch_size", "grid_size", "exif_transpose"}:
        raise ValueError("Preprocessing requires patch_size, grid_size, exif_transpose")
    for key in ("patch_size", "grid_size"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["exif_transpose"]) is not bool:
        raise ValueError("exif_transpose must be boolean")


def validate_dataset(dataset):
    required = {"file_path", "subject_id", "noise", "severity", "route"}
    if not required.issubset(dataset.columns) or dataset.empty:
        raise ValueError(f"Dataset needs nonempty columns {sorted(required)}")
    frame = dataset.copy()
    if frame[list(required)].isna().any().any():
        raise ValueError("Dataset contains missing values")
    for col in required:
        frame[col] = frame[col].astype(str)
    canonical = frame.file_path.map(lambda p: str(Path(p).resolve()))
    if canonical.duplicated().any() or frame.duplicated(["subject_id", "severity"]).any():
        raise ValueError("Duplicate image path or person/severity")
    for row in frame.itertuples():
        if not Path(row.file_path).is_file():
            raise ValueError(f"Missing image: {row.file_path}")
        validate_prediction({key: getattr(row, key) for key in ("noise", "severity", "route")})
    if any(set(group.severity) != {"0", "1", "2", "3"} or len(group) != 4
           for _, group in frame.groupby("subject_id")):
        raise ValueError("Each person must have exactly one image per severity 0–3")
    return frame


def create_subject_split(dataset, seed=42, *, project_root=None, counts=(35, 5, 10),
                         fold_index=None, folds=5):
    """Save portable membership, labels and content hashes (default 50 people)."""
    frame = validate_dataset(dataset)
    root = Path(project_root or Path.cwd()).resolve()
    if len(counts) != 3 or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError("counts must contain three positive person counts")
    ids = sorted(frame.subject_id.unique())
    if len(ids) != sum(counts):
        raise ValueError(f"Expected {sum(counts)} people / {4 * sum(counts)} images; found {len(ids)} people")
    shuffled = np.random.default_rng(seed).permutation(ids).tolist()
    manifest = {"schema_version": 1, "seed": seed, "counts": list(counts), "partitions": {}}
    if fold_index is not None:
        if (type(folds) is not int or folds < 2 or type(fold_index) is not int or
                not 0 <= fold_index < folds or counts[2] * folds != len(ids)):
            raise ValueError("Grouped folds require a valid fold index and equal test groups covering all subjects")
        # Fold zero exactly preserves the original split. Rotating the seeded
        # permutation makes each subject a test subject once across the folds.
        offset = counts[2] * fold_index
        shuffled = shuffled[offset:] + shuffled[:offset]
        manifest.update(schema_version=2, fold_index=fold_index, folds=folds)
    offset = 0
    for partition, count in zip(PARTITIONS, counts):
        subjects = shuffled[offset:offset + count]
        offset += count
        records = []
        for row in frame.loc[frame.subject_id.isin(subjects)].itertuples():
            path = Path(row.file_path).resolve()
            records.append({"file_path": path.relative_to(root).as_posix(), "subject_id": row.subject_id,
                            **{key: getattr(row, key) for key in ("noise", "severity", "route")},
                            "sha256": file_sha256(path)})
        manifest["partitions"][partition] = {"subject_ids": subjects, "records": records}
    return manifest


def validate_subject_split(manifest, dataset, project_root):
    """Reject changed labels/content, incomplete membership and partition overlap."""
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported split schema version")
    fold_args = ({"fold_index": manifest["fold_index"], "folds": manifest["folds"]}
                 if manifest["schema_version"] == 2 else {})
    expected = create_subject_split(dataset, manifest["seed"], project_root=project_root,
                                    counts=manifest["counts"], **fold_args)
    # Saved row ordering is authoritative and may differ from the current frame.
    def normalized(value):
        value = json.loads(json.dumps(value))
        for partition in PARTITIONS:
            value["partitions"][partition]["records"].sort(key=lambda row: row["file_path"])
        return value
    if normalized(expected) != normalized(manifest):
        raise ValueError("Split membership, labels or image hashes changed")
    return manifest


def load_subject_split(path, dataset, project_root):
    return validate_subject_split(json.loads(Path(path).read_text()), dataset, project_root)


def partition_frame(manifest, partition, project_root):
    frame = pd.DataFrame(manifest["partitions"][partition]["records"])
    root = Path(project_root).resolve()
    paths = []
    for value in frame.file_path:
        path = (root / value).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest paths must stay within the project root")
        paths.append(str(path))
    frame["file_path"] = paths
    return frame


def load_image_patches(image_path, preprocess_config):
    validate_preprocessing(preprocess_config)
    size, grid = preprocess_config["patch_size"], preprocess_config["grid_size"]
    with Image.open(image_path) as image:
        if preprocess_config["exif_transpose"]:
            image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        width, height = image.size
        if min(width, height) < size:
            raise ValueError(f"Image {image_path} is smaller than patch_size={size}")
        patches = []
        fractions = [(i + 1) / (grid + 1) for i in range(grid)]
        for fy in fractions:
            top = max(0, min(height - size, math.floor(fy * (height - 1) - size / 2)))
            for fx in fractions:
                left = max(0, min(width - size, math.floor(fx * (width - 1) - size / 2)))
                pixels = np.array(image.crop((left, top, left + size, top + size)), dtype=np.float32)
                patches.append(torch.from_numpy(pixels).permute(2, 0, 1) / 255.0)
    return torch.stack(patches)


def validate_image_sizes(dataset, preprocess_config):
    validate_preprocessing(preprocess_config)
    for path in dataset.file_path:
        with Image.open(path) as image:
            if min(image.size) < preprocess_config["patch_size"]:
                raise ValueError(f"Image {path} is smaller than patch_size")


class FaceSeverityDataset(Dataset):
    def __init__(self, frame, preprocess_config):
        validate_preprocessing(preprocess_config)
        self.frame = frame.reset_index(drop=True).copy()
        self.preprocess_config = dict(preprocess_config)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return load_image_patches(row.file_path, self.preprocess_config), int(row.severity)
