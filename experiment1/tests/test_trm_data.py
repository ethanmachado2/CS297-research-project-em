import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from src.trm_data import (create_subject_split, load_subject_split, partition_frame,
                          load_image_patches, validate_dataset, validate_preprocessing)
from trm_fixtures import make_dataset


class TRMDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.frame = make_dataset(self.root)
        self.prep = {"patch_size": 4, "grid_size": 3, "exif_transpose": False}

    def test_default_split_counts_and_disjoint_people(self):
        frame = make_dataset(self.root / "fifty", people=50)
        split = create_subject_split(frame, project_root=self.root)
        people, paths = set(), set()
        for partition, count in zip(("train", "validation", "test"), (35, 5, 10)):
            part = split["partitions"][partition]
            self.assertEqual(len(part["subject_ids"]), count)
            self.assertFalse(people.intersection(part["subject_ids"]))
            people.update(part["subject_ids"])
            self.assertEqual(len(part["records"]), 4 * count)
            for severity in range(4):
                self.assertEqual(sum(r["severity"] == str(severity) for r in part["records"]), count)
            for record in part["records"]:
                self.assertNotIn(record["file_path"], paths)
                self.assertFalse(Path(record["file_path"]).is_absolute())
                paths.add(record["file_path"])

    def test_deterministic_membership_preserves_frame_order(self):
        split = create_subject_split(self.frame, project_root=self.root, counts=(2, 1, 1))
        reverse = self.frame.iloc[::-1]
        other = create_subject_split(reverse, project_root=self.root, counts=(2, 1, 1))
        for part in split["partitions"]:
            self.assertEqual(split["partitions"][part]["subject_ids"], other["partitions"][part]["subject_ids"])
            self.assertEqual(split["partitions"][part]["records"], other["partitions"][part]["records"][::-1])
        path = self.root / "split.json"
        path.write_text(json.dumps(split))
        self.assertEqual(load_subject_split(path, reverse, self.root), split)

    def test_invalid_dataset_groups_paths_and_labels(self):
        invalid = [self.frame.iloc[:-1], self.frame.iloc[[*range(len(self.frame)), 0]]]
        for column, value in (("file_path", "/missing/file.png"), ("severity", "9"),
                              ("noise", "0"), ("subject_id", None)):
            frame = self.frame.copy()
            # Row severity 1 makes noise=0 an invalid label mapping.
            frame.loc[frame.severity == "1", column] = value
            invalid.append(frame)
        frame = self.frame.copy()
        frame.loc[1, "file_path"] = frame.loc[0, "file_path"]
        invalid.append(frame)
        for frame in invalid:
            with self.subTest(frame=frame.shape), self.assertRaises((ValueError, TypeError)):
                validate_dataset(frame)
        with self.assertRaises(ValueError):
            create_subject_split(self.frame, project_root=self.root)

    def test_manifest_relocation_changes_and_overlap(self):
        split = create_subject_split(self.frame, project_root=self.root, counts=(2, 1, 1))
        path = self.root / "split.json"
        path.write_text(json.dumps(split))
        with tempfile.TemporaryDirectory() as moved:
            moved = Path(moved).resolve()
            shutil.copytree(self.root / "data", moved / "data")
            frame = self.frame.copy()
            frame.file_path = frame.file_path.map(lambda p: str(moved / Path(p).relative_to(self.root)))
            self.assertEqual(load_subject_split(path, frame, moved), split)
            changed = Path(frame.file_path.iloc[0])
            changed.write_bytes(changed.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_subject_split(path, frame, moved)
        bad = copy.deepcopy(split)
        bad["partitions"]["test"]["subject_ids"] = bad["partitions"]["train"]["subject_ids"]
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "changed"):
            load_subject_split(path, self.frame, self.root)

    def test_exact_native_crops_channel_scale_and_order(self):
        y, x = np.indices((20, 24))
        pixels = np.stack((x, y, x + y), axis=-1).astype(np.uint8)
        path = self.root / "coordinates.png"
        Image.fromarray(pixels).save(path)
        patches = load_image_patches(path, self.prep)
        self.assertEqual(patches.shape, (9, 3, 4, 4))
        self.assertEqual(patches.dtype, torch.float32)
        for i, (top, left) in enumerate((t, l) for t in (2, 7, 12) for l in (3, 9, 15)):
            expected = torch.from_numpy(pixels[top:top + 4, left:left + 4].copy()).permute(2, 0, 1).float() / 255
            torch.testing.assert_close(patches[i], expected)

    def test_color_conversion_small_corrupt_and_no_exif_rotation(self):
        for mode in ("L", "RGBA"):
            path = self.root / f"{mode}.png"
            Image.new(mode, (12, 12)).save(path)
            self.assertEqual(load_image_patches(path, self.prep).shape, (9, 3, 4, 4))
        small = self.root / "small.png"
        Image.new("RGB", (2, 2)).save(small)
        with self.assertRaisesRegex(ValueError, "smaller"):
            load_image_patches(small, self.prep)
        small.write_text("corrupt")
        with self.assertRaises(OSError):
            load_image_patches(small, self.prep)
        image = Image.new("RGB", (20, 30))
        for x in range(10):
            for y in range(30):
                image.putpixel((x, y), (255, 0, 0))
        exif = image.getexif()
        exif[274] = 6
        path = self.root / "rotated.jpg"
        image.save(path, exif=exif)
        raw = load_image_patches(path, self.prep)
        rotated = load_image_patches(path, dict(self.prep, exif_transpose=True))
        self.assertFalse(torch.equal(raw, rotated))

    def test_pixels_do_not_depend_on_path_or_labels(self):
        original = Path(self.frame.file_path.iloc[0])
        copied = self.root / "unrelated" / "severity3.png"
        copied.parent.mkdir()
        shutil.copyfile(original, copied)
        torch.testing.assert_close(load_image_patches(original, self.prep), load_image_patches(copied, self.prep))

    def test_invalid_preprocessing_and_traversal(self):
        for key, value in (("patch_size", 0), ("grid_size", -1), ("exif_transpose", "false")):
            with self.assertRaises(ValueError):
                validate_preprocessing(dict(self.prep, **{key: value}))
        split = create_subject_split(self.frame, project_root=self.root, counts=(2, 1, 1))
        split["partitions"]["test"]["records"][0]["file_path"] = "../outside.png"
        with self.assertRaises(ValueError):
            partition_frame(split, "test", self.root)
