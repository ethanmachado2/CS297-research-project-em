"""Cloud Storage staging, periodic artifact uploads, and run provenance.

Google Cloud imports are lazy so local runs and tests need no credentials.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import threading
import warnings


def parse_gcs_uri(uri):
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError("Expected a gs://bucket/prefix URI")
    bucket, _, prefix = uri[5:].partition("/")
    prefix = prefix.strip("/")
    if not bucket or not prefix or any(part in {".", ".."} for part in prefix.split("/")):
        raise ValueError("Use a bucket and a nonempty, safe prefix: gs://bucket/prefix")
    return bucket, prefix


def storage_client():
    from google.cloud import storage
    return storage.Client()  # Application Default Credentials; never embed keys.


def download_gcs_tree(uri, destination, *, client=None):
    bucket, prefix = parse_gcs_uri(uri)
    client = client if client is not None else storage_client()
    destination = Path(destination).resolve()
    count = 0
    for blob in client.list_blobs(bucket, prefix=prefix + "/"):
        if blob.name.endswith("/"):
            continue
        relative = blob.name.removeprefix(prefix + "/")
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ".." in parts.parts or "\\" in relative:
            raise ValueError(f"Unsafe Cloud Storage object path: {blob.name}")
        path = destination.joinpath(*parts.parts)
        if not path.resolve().is_relative_to(destination):
            raise ValueError(f"Cloud Storage object escapes destination: {blob.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(path))
        count += 1
    if not count:
        raise ValueError(f"No files found under {uri}")
    return destination


def download_gcs_file(uri, destination, *, client=None):
    bucket, name = parse_gcs_uri(uri)
    client = client if client is not None else storage_client()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    client.bucket(bucket).blob(name).download_to_filename(str(temporary))
    temporary.replace(destination)
    return destination


class ArtifactSync:
    """Upload changed run files periodically and on normal/error context exit.

    Each checkpoint is opened once while uploading: atomic replacements in the
    trainer cannot switch the underlying file midway through an upload. The
    .tmp files used by the trainers are excluded. A killed VM can still lose
    updates since the last completed sync; this is not training resume.
    """
    def __init__(self, directory, gcs_uri=None, *, interval=60, client=None):
        if interval <= 0:
            raise ValueError("Upload interval must be positive")
        self.directory = Path(directory)
        self.uri = gcs_uri
        self.interval = interval
        self.client = client
        self.uploaded = {}
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        if gcs_uri:
            self.bucket_name, self.prefix = parse_gcs_uri(gcs_uri)

    def sync(self):
        if not self.uri:
            return
        with self.lock:
            bucket = self.client.bucket(self.bucket_name)
            for path in sorted(self.directory.rglob("*")):
                if not path.is_file() or path.is_symlink() or path.suffix == ".tmp":
                    continue
                relative = path.relative_to(self.directory).as_posix()
                # Hash metadata from the open descriptor, not a later replacement.
                with path.open("rb") as stream:
                    stat = os.fstat(stream.fileno())
                    signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
                    if self.uploaded.get(relative) == signature:
                        continue
                    bucket.blob(self.prefix + "/" + relative).upload_from_file(stream)
                self.uploaded[relative] = signature

    def _periodic(self):
        while not self.stop.wait(self.interval):
            try:
                self.sync()
            except Exception as exc:
                warnings.warn(f"Periodic artifact upload failed; will retry: {exc}", RuntimeWarning)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write_status("running")
        if self.uri:
            if self.client is None:
                self.client = storage_client()
            self.sync()  # Fail before expensive work if credentials/access are invalid.
            self.thread = threading.Thread(target=self._periodic, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        if self.thread:
            self.thread.join()
        status = "completed" if exc_type is None else (
            "interrupted" if issubclass(exc_type, (KeyboardInterrupt, SystemExit)) else "failed")
        self._write_status(status, exc_type.__name__ if exc_type else None)
        try:
            self.sync()
        except Exception as upload_error:
            if exc_type is None:
                raise
            warnings.warn(f"Final upload failed; local artifacts remain at {self.directory}: "
                          f"{upload_error}", RuntimeWarning)
        return False

    def _write_status(self, status, error_type=None):
        value = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(),
                 "gcs_uri": self.uri, "error_type": error_type}
        temporary = self.directory / "cloud_status.json.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(self.directory / "cloud_status.json")


def write_run_provenance(session, project_root, dataset_manifest, settings):
    session, project = Path(session), Path(project_root)
    session.mkdir(parents=True, exist_ok=True)

    def git(*args):
        result = subprocess.run(["git", "-C", str(project), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")}
    provenance = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
                  "python": sys.version, "platform": platform.platform(), "git_commit": git("rev-parse", "HEAD"),
                  "git_dirty": bool(git("status", "--porcelain")), "settings": settings,
                  "packages": dict(sorted(packages.items()))}
    source = project / "data/dataset_source.json"
    if source.exists():
        recorded = json.loads(source.read_text())
        if recorded.get("content_digest") == dataset_manifest["content_digest"]:
            provenance["dataset_source"] = recorded["source"]
    (session / "environment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (session / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2) + "\n")
