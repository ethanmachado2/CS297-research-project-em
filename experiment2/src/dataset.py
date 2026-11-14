"""Download from Kaggle once; validate and reuse the cached images in place."""
import json
from pathlib import Path

from PIL import Image

from .image_quality_pipeline import build_dataset
from .trm_data import file_sha256, manifest_digest, validate_dataset

HANDLE = "trainingdatapro/human-faces-augmentation-image-restoration"


def inspect_dataset(root, expected_subjects=50):
    root = Path(root).resolve()
    frame = validate_dataset(build_dataset(root))
    if len(frame) != expected_subjects * 4:
        raise ValueError(f"Expected {expected_subjects * 4} images, found {len(frame)} under {root}")
    records = []
    for row in frame.itertuples():
        path = Path(row.file_path)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if min(width, height) < 128:
                raise ValueError(f"Image too small for TRM patches: {path}")
        records.append({"path": path.relative_to(root).as_posix(), "subject_id": row.subject_id,
                        "severity": row.severity, "width": width, "height": height,
                        "sha256": file_sha256(path)})
    return {"images": len(frame), "subjects": expected_subjects, "records": records,
            "content_digest": manifest_digest(records)}


def find_dataset_root(root):
    root = Path(root).resolve()
    candidates = [root, *sorted(p.parent for p in root.rglob("images") if p.is_dir())]
    valid = []
    for candidate in dict.fromkeys(candidates):
        if not (candidate / "augmentation").is_dir():
            continue
        try:
            manifest = inspect_dataset(candidate)
        except (ValueError, OSError):
            continue
        valid.append((candidate, manifest))
    if len(valid) != 1:
        raise ValueError(f"Expected one complete 200-image dataset under {root}; found {len(valid)}")
    return valid[0]


def ensure_dataset(project, *, dataset_root=None):
    """Use an explicit root, a validated previous download, or KaggleHub cache.

    An explicit invalid root is an error. A missing or invalid cached root is
    reacquired from Kaggle. No dataset bytes are copied into the repository.
    """
    source_file = Path(project) / "data/dataset_source.json"
    if dataset_root is not None:
        root, manifest = find_dataset_root(dataset_root)
        source = {"kind": "explicit", "root": str(root)}
    else:
        source = {}
        if source_file.is_file():
            try:
                previous = json.loads(source_file.read_text())
                root = Path(previous["root"])
                manifest = inspect_dataset(root)
                if manifest["content_digest"] == previous["content_digest"]:
                    print(f"Reusing validated dataset: {root}", flush=True)
                    return root, manifest, previous
            except (OSError, ValueError, KeyError):
                pass
        import kagglehub
        downloaded = Path(kagglehub.dataset_download(HANDLE))
        root, manifest = find_dataset_root(downloaded)
        source = {"kind": "kagglehub", "handle": HANDLE, "download_root": str(downloaded),
                  "root": str(root)}
        if downloaded.parent.name == "versions" and downloaded.name.isdigit():
            source["resolved_version"] = int(downloaded.name)
    source["content_digest"] = manifest["content_digest"]
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(json.dumps(source, indent=2) + "\n")
    print(f"Validated {manifest['images']} images: {root}", flush=True)
    return root, manifest, source
