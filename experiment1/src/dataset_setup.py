"""Validate or prepare the experiment dataset without loading a neural model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from PIL import Image

from .image_quality_pipeline import build_dataset

DEFAULT_KAGGLE_HANDLE = "trainingdatapro/human-faces-augmentation-image-restoration"


def inspect_dataset(root, *, expected_subjects=50, min_image_size=128):
    """Check all four folders, subject membership, decodable pixels, and hashes."""
    root = Path(root).resolve()
    frame = build_dataset(root)
    if len(frame) != expected_subjects * 4:
        raise ValueError(f"Expected {expected_subjects * 4} images under {root}; found {len(frame)}")
    counts = frame.groupby(["subject_id", "severity"]).size().unstack(fill_value=0)
    if (len(counts) != expected_subjects or set(counts.columns) != {"0", "1", "2", "3"}
            or not counts.eq(1).all().all()):
        raise ValueError("Expected exactly one image of every severity for each subject")
    records = []
    for row in frame.itertuples():
        path = Path(row.file_path)
        with Image.open(path) as image:
            image.load()
            if min(image.size) < min_image_size:
                raise ValueError(f"Image must be at least {min_image_size}px in each dimension: {path}")
        records.append({"path": path.relative_to(root).as_posix(), "subject_id": row.subject_id,
                        "severity": row.severity, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"schema_version": 1, "subjects": expected_subjects, "images": len(frame),
            "content_digest": digest, "records": records}


def find_dataset_root(download_root, **validation):
    """Handle Kaggle archives with a files/ or data/files/ wrapper."""
    root = Path(download_root)
    candidates = [root, *sorted(path.parent for path in root.rglob("images") if path.is_dir())]
    valid = []
    for candidate in dict.fromkeys(candidates):
        if not (candidate / "images").is_dir() or not (candidate / "augmentation").is_dir():
            continue
        try:
            inspect_dataset(candidate, **validation)
        except (ValueError, OSError):
            continue
        valid.append(candidate)
    if len(valid) != 1:
        raise ValueError(f"Expected one complete images/ + augmentation/ dataset under {root}; "
                         f"found {len(valid)}. Check the archive layout and selected dataset version.")
    return valid[0]


def ensure_dataset(project_root, *, kaggle_handle=DEFAULT_KAGGLE_HANDLE, kaggle_version=None,
                   gcs_uri=None, expected_subjects=50, min_image_size=128, return_manifest=False):
    """Reuse valid local data; otherwise stage, validate, and promote a download.

    An incomplete existing directory is preserved as files.incomplete.<uuid>.
    Authentication/download failures never replace existing data. GCS, when
    selected, is authoritative: access errors do not silently fall back to Kaggle.
    Set return_manifest=True to receive (path, manifest) without decoding all
    images a second time to collect run metadata.
    """
    project = Path(project_root).resolve()
    target = project / "data/files"
    validation = {"expected_subjects": expected_subjects, "min_image_size": min_image_size}
    try:
        manifest = inspect_dataset(target, **validation)
        print(f"Reusing validated dataset: {target}")
        return (target, manifest) if return_manifest else target
    except (ValueError, OSError):
        pass
    if not gcs_uri and not kaggle_handle:
        raise ValueError("Dataset is incomplete. Configure a Kaggle handle or a gs:// input prefix.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataset-stage-", dir=target.parent) as temporary:
        stage = Path(temporary)
        source = {"gcs_uri": gcs_uri} if gcs_uri else {"kaggle_handle": kaggle_handle}
        if gcs_uri:
            from .cloud_runtime import download_gcs_tree
            download_gcs_tree(gcs_uri, stage / "download")
            downloaded = stage / "download"
        else:
            import kagglehub
            handle = kaggle_handle
            if kaggle_version is not None:
                if not str(kaggle_version).isdigit() or int(kaggle_version) < 1:
                    raise ValueError("Kaggle version must be a positive integer")
                if "/versions/" in handle:
                    raise ValueError("Set the version in the handle or kaggle_version, not both")
                handle += f"/versions/{kaggle_version}"
            downloaded = Path(kagglehub.dataset_download(handle))
            source["requested_handle"] = handle
            # KaggleHub's cache contains an immutable versions/<number> directory.
            if downloaded.parent.name == "versions" and downloaded.name.isdigit():
                source["resolved_version"] = int(downloaded.name)
        dataset_root = find_dataset_root(downloaded, **validation)
        prepared = stage / "prepared"
        shutil.copytree(dataset_root, prepared)
        manifest = inspect_dataset(prepared, **validation)
        backup = target.with_name(f"files.incomplete.{uuid4().hex[:8]}")
        had_target = target.exists()
        if had_target:
            target.rename(backup)
        try:
            prepared.rename(target)
        except BaseException:
            if had_target:
                backup.rename(target)
            raise
        provenance = {"source": source, "content_digest": manifest["content_digest"]}
        (target.parent / "dataset_source.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if had_target:
            print(f"Preserved incomplete dataset: {backup}")
    print(f"Prepared and validated dataset: {target}")
    return (target, manifest) if return_manifest else target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--kaggle-handle", default=DEFAULT_KAGGLE_HANDLE)
    parser.add_argument("--kaggle-version", type=int)
    parser.add_argument("--gcs-uri")
    args = parser.parse_args()
    ensure_dataset(args.project_root, kaggle_handle=args.kaggle_handle,
                   kaggle_version=args.kaggle_version, gcs_uri=args.gcs_uri)


if __name__ == "__main__":
    main()
