import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src.cloud_runtime import ArtifactSync, download_gcs_file, download_gcs_tree, parse_gcs_uri
from src.dataset_setup import ensure_dataset, inspect_dataset

PROJECT = Path(__file__).resolve().parents[1]


def make_images(root):
    root = Path(root)
    for severity in range(4):
        folder = root / ("images" if severity == 0 else f"augmentation/{severity}")
        folder.mkdir(parents=True, exist_ok=True)
        for subject in range(2):
            Image.new("RGB", (16, 16), (severity * 50, subject * 50, 0)).save(folder / f"{subject}.png")
    return root


class DatasetSetupTests(unittest.TestCase):
    def test_valid_existing_data_needs_no_download_and_hashes_are_portable(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root = make_images(project / "data/files")
            with patch.dict("sys.modules", {"kagglehub": Mock()}) as modules:
                result, manifest = ensure_dataset(project, expected_subjects=2, min_image_size=16,
                                                  return_manifest=True)
                modules["kagglehub"].dataset_download.assert_not_called()
            self.assertEqual(result, root.resolve())
            self.assertEqual(manifest["images"], 8)
            self.assertTrue(all(not Path(row["path"]).is_absolute() for row in manifest["records"]))
            Image.new("RGB", (16, 16), "white").save(root / "images/0.png")
            changed = inspect_dataset(root, expected_subjects=2, min_image_size=16)
            self.assertNotEqual(manifest["content_digest"], changed["content_digest"])

    def test_nested_kaggle_download_preserves_incomplete_data_and_records_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            original = project / "data/files"
            original.mkdir(parents=True)
            (original / "keep.txt").write_text("incomplete data")
            cache = Path(temporary) / "cache/versions/3"
            make_images(cache / "wrapper/files")
            downloader = Mock(return_value=str(cache))
            with patch.dict("sys.modules", {"kagglehub": SimpleNamespace(dataset_download=downloader)}):
                ensure_dataset(project, kaggle_version=3, expected_subjects=2, min_image_size=16)
            self.assertTrue(next(original.parent.glob("files.incomplete.*/keep.txt")).is_file())
            source = json.loads((project / "data/dataset_source.json").read_text())
            self.assertEqual(source["source"]["resolved_version"], 3)
            self.assertTrue(downloader.call_args.args[0].endswith("/versions/3"))

    def test_failed_or_invalid_download_does_not_replace_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            original = project / "data/files"
            original.mkdir(parents=True)
            marker = original / "keep.txt"
            marker.write_text("keep")
            bad_cache = project / "invalid-download"
            bad_cache.mkdir()
            downloader = Mock(side_effect=PermissionError("authentication required"))
            with patch.dict("sys.modules", {"kagglehub": SimpleNamespace(dataset_download=downloader)}):
                with self.assertRaises(PermissionError):
                    ensure_dataset(project, expected_subjects=2, min_image_size=16)
                downloader.side_effect = None
                downloader.return_value = str(bad_cache)
                with self.assertRaisesRegex(ValueError, "archive layout"):
                    ensure_dataset(project, expected_subjects=2, min_image_size=16)
            self.assertEqual(marker.read_text(), "keep")
            self.assertEqual(list(original.parent.glob("files.incomplete.*")), [])

    def test_bad_subject_membership_and_corrupt_pixels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = make_images(temporary)
            (root / "augmentation/3/1.png").rename(root / "augmentation/3/other.png")
            with self.assertRaisesRegex(ValueError, "every severity"):
                inspect_dataset(root, expected_subjects=2, min_image_size=16)
            (root / "augmentation/3/other.png").rename(root / "augmentation/3/1.png")
            (root / "images/0.png").write_bytes(b"not an image")
            with self.assertRaises(OSError):
                inspect_dataset(root, expected_subjects=2, min_image_size=16)


class CloudStorageTests(unittest.TestCase):
    def test_uri_and_download_path_validation(self):
        self.assertEqual(parse_gcs_uri("gs://bucket/runs/"), ("bucket", "runs"))
        for uri in ["bucket/runs", "gs://bucket", "gs:///runs", "gs://bucket/../runs"]:
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                parse_gcs_uri(uri)
        with tempfile.TemporaryDirectory() as temporary:
            client = Mock()
            client.list_blobs.return_value = [SimpleNamespace(name="inputs/../../escape")]
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                download_gcs_tree("gs://bucket/inputs", temporary, client=client)
            client.list_blobs.return_value = []
            with self.assertRaisesRegex(ValueError, "No files"):
                download_gcs_tree("gs://bucket/inputs", temporary, client=client)

    def test_gcs_dataset_source_is_used_without_kaggle_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            def stage(uri, destination):
                make_images(destination)
            with patch("src.cloud_runtime.download_gcs_tree", side_effect=stage) as download:
                ensure_dataset(temporary, gcs_uri="gs://bucket/inputs", expected_subjects=2, min_image_size=16)
            download.assert_called_once()
            source = json.loads((Path(temporary) / "data/dataset_source.json").read_text())
            self.assertEqual(source["source"], {"gcs_uri": "gs://bucket/inputs"})

    def test_changed_artifacts_finally_upload_on_training_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = Mock()
            uploads = []
            bucket = client.bucket.return_value
            def blob(name):
                return SimpleNamespace(upload_from_file=lambda stream: uploads.append((name, stream.read())))
            bucket.blob.side_effect = blob
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with ArtifactSync(root, "gs://bucket/runs/one", client=client) as sync:
                    (root / "best.pt").write_bytes(b"checkpoint")
                    (root / "best.tmp").write_bytes(b"partial")
                    sync.sync()
                    before = len(uploads)
                    sync.sync()
                    self.assertEqual(len(uploads), before)
                    raise RuntimeError("training failed")
            self.assertIn(("runs/one/best.pt", b"checkpoint"), uploads)
            self.assertFalse(any(name.endswith("best.tmp") for name, _ in uploads))
            self.assertEqual(json.loads(uploads[-1][1])["status"], "failed")
            self.assertFalse(sync.thread.is_alive())

    def test_final_upload_failure_is_reported_and_local_mode_needs_no_client(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("src.cloud_runtime.storage_client", side_effect=AssertionError("no network")):
                with ArtifactSync(temporary):
                    pass
            self.assertEqual(json.loads((Path(temporary) / "cloud_status.json").read_text())["status"], "completed")
            client = Mock()
            with self.assertRaisesRegex(OSError, "upload failed"):
                with ArtifactSync(temporary, "gs://bucket/runs", client=client) as sync:
                    client.bucket.side_effect = OSError("upload failed")

    def test_periodic_upload_happens_before_context_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            uploaded = threading.Event()
            client = Mock()
            def blob(name):
                def upload(stream):
                    if name.endswith("/best.pt") and stream.read() == b"weights":
                        uploaded.set()
                return SimpleNamespace(upload_from_file=upload)
            client.bucket.return_value.blob.side_effect = blob
            with ArtifactSync(temporary, "gs://bucket/runs", interval=0.01, client=client):
                (Path(temporary) / "best.pt").write_bytes(b"weights")
                self.assertTrue(uploaded.wait(timeout=3), "Background upload did not run")

    def test_checkpoint_download_preserves_old_file_until_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.pt"
            path.write_bytes(b"old")
            client = Mock()
            blob = client.bucket.return_value.blob.return_value
            blob.download_to_filename.side_effect = OSError("network failure")
            with self.assertRaises(OSError):
                download_gcs_file("gs://bucket/checkpoints/best.pt", path, client=client)
            self.assertEqual(path.read_bytes(), b"old")
            blob.download_to_filename.side_effect = lambda name: Path(name).write_bytes(b"new")
            self.assertEqual(download_gcs_file("gs://bucket/checkpoints/best.pt", path, client=client), path)
            self.assertEqual(path.read_bytes(), b"new")


class CloudInstallerTests(unittest.TestCase):
    def test_cloud_install_uses_single_requirements_and_preserves_managed_stack(self):
        spec = importlib.util.spec_from_file_location("installer", PROJECT / "scripts/install_cloud_dependencies.py")
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        result = installer.cloud_requirements((PROJECT / "requirements.txt").read_text())
        self.assertNotIn("torch==", result)
        self.assertNotIn("torchvision==", result)
        self.assertNotIn("jupyter==", result)
        self.assertIn("transformers==", result)
        self.assertIn("kagglehub", result)
        self.assertFalse((PROJECT / "requirements-trm.txt").exists())


if __name__ == "__main__":
    unittest.main()
