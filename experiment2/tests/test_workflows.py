"""Offline checks; CUDA tests exercise the actual AMP training path if present."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.attention import disable_native_gqa, validate_efficient_attention
from src.dataset import ensure_dataset, inspect_dataset
from src.image_quality_pipeline import (build_dataset, parse_prediction, prepare_few_shot_split,
                                        run_inference, FEW_SHOT_GUIDANCE)
from src.run import select_qwen_targets
from src.trm_data import create_subject_split, partition_frame, validate_subject_split
from src.trm_model import VisionTRM, PROVENANCE_PATH
from src.trm_reporting import export_saved_run
from src.trm_training import train_experiment, load_checkpoint
from src.qwen_adapter import QwenImageModel

PROJECT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def make_images(root, people=50):
    for severity in range(4):
        folder = root / ("images" if severity == 0 else f"augmentation/{severity}")
        folder.mkdir(parents=True, exist_ok=True)
        for subject in range(people):
            Image.new("RGB", (128, 128), color=(subject, severity * 50, 20)).save(folder / f"{subject}.jpg")
    return build_dataset(root)


def tiny_config():
    config = json.loads((PROJECT / "configs/trm.json").read_text())
    config["split"]["counts"] = [2, 1, 1]
    config["preprocessing"].update(patch_size=8, grid_size=1)
    config["model"].update(hidden_size=16, num_heads=2, L_layers=2, H_cycles=2, L_cycles=1,
                           refinement_steps=2)
    config["training"].update(batch_size=3, max_epochs=2, patience=2)
    return config


class DataAndReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def test_kaggle_download_then_validated_cache_reuse_without_copy(self):
        data = self.root / "mounted/files"
        make_images(data)
        downloader = types.SimpleNamespace(dataset_download=lambda handle: str(data.parent))
        project = self.root / "project"
        with patch.dict(sys.modules, {"kagglehub": downloader}):
            root, manifest, _ = ensure_dataset(project)
        with patch.dict(sys.modules, {"kagglehub": None}):
            cached, again, _ = ensure_dataset(project)
        self.assertEqual(root, data)
        self.assertEqual(cached, data)
        self.assertEqual(manifest["content_digest"], again["content_digest"])
        self.assertEqual(manifest["images"], 200)
        self.assertFalse((project / "data/files").exists())

    def test_invalid_explicit_dataset_and_changed_split_are_rejected(self):
        data = self.root / "imageset"
        frame = make_images(data, people=4)
        with self.assertRaises(ValueError):
            ensure_dataset(self.root / "project", dataset_root=data)
        split = create_subject_split(frame, project_root=data, counts=[2, 1, 1])
        self.assertEqual(validate_subject_split(split, frame, data), split)
        partitions = [set(value["subject_ids"]) for value in split["partitions"].values()]
        self.assertFalse(partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2])
        Image.new("RGB", (128, 128), "red").save(frame.file_path.iloc[0])
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_subject_split(split, frame, data)

    def test_paired_qwen_selection_and_reference_exclusion(self):
        frame = make_images(self.root)
        config = json.loads((PROJECT / "configs/qwen.json").read_text())
        full, examples = select_qwen_targets(frame, config, "full")
        smoke, _ = select_qwen_targets(frame, config, "smoke")
        largest, _ = select_qwen_targets(frame, config, "largest")
        self.assertEqual(len(full), 196)
        self.assertEqual(len(smoke), 12)
        self.assertEqual(smoke.subject_id.tolist(), ["27"] * 4 + ["14"] * 4 + ["9"] * 4)
        self.assertEqual(largest.severity.tolist(), ["3"])
        self.assertEqual(largest.subject_id.tolist(), ["18"])
        self.assertNotIn("0", full.subject_id.tolist())
        self.assertEqual([e.prediction["severity"] for e in examples], ["0", "1", "2", "3"])

    def test_strict_failures_stay_in_planned_accuracy_and_raw_output(self):
        frame = make_images(self.root / "dataset", people=1).iloc[:2]
        class Model:
            def generate(self, path, prompt):
                if Path(path).parent.name == "images":
                    return '{"noise":"0","severity":"0","route":"0"}'
                return '```json\n{"noise":"1","severity":"1","route":"0"}\n```'
        output = self.root / "run"
        run_inference(Model(), frame.file_path, labels=frame, output_dir=output,
                      experiment_number=2, run_number=1, show_progress=False)
        report = export_saved_run(output / "experiment_002/run_001")
        self.assertEqual(len(report["failures"]), 1)
        self.assertEqual(report["summary"].loc["severity", "Accuracy (valid)"], 1.0)
        self.assertEqual(report["summary"].loc["severity", "Correct / planned"], 0.5)
        self.assertTrue(report["failures"].raw_output.iloc[0].startswith("```json"))
        with self.assertRaises(FileExistsError):
            run_inference(Model(), [], output_dir=output, experiment_number=2, run_number=1)


class ModelTests(unittest.TestCase):
    def test_qwen_original_images_and_reference_turns_are_preserved(self):
        class Inputs(dict):
            def to(self, device):
                return self
        class Processor:
            image_processor = types.SimpleNamespace(
                size={"shortest_edge": 65536, "longest_edge": 16777216}, do_resize=True)
            def to_dict(self):
                return {}
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.image_modes = [item["image"].mode for turn in messages for item in turn["content"]
                                    if item["type"] == "image"]
                return Inputs(input_ids=torch.zeros(1, 3, dtype=torch.long),
                              image_grid_thw=torch.ones(5, 3, dtype=torch.long))
            def decode(self, tokens, **kwargs):
                return '{"noise":"1","severity":"2","route":"1"}'
        class Model(torch.nn.Module):
            device = torch.device("cpu")
            dtype = torch.float16
            def __init__(self):
                super().__init__()
                self.lm_head = torch.nn.Linear(2, 2)
            def generate(self, **kwargs):
                self.settings = kwargs
                return torch.zeros(1, 5, dtype=torch.long)
        # Message assembly is tested without invoking a model or a CUDA kernel.
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory() as temporary:
            frame = make_images(Path(temporary), people=2)
            targets, examples = prepare_few_shot_split(frame)
            processor, model = Processor(), Model()
            adapter = QwenImageModel(model, processor, model_name="fixture", snapshot="revision",
                                     mode="few_shot", example_subject="0")
            with patch("src.qwen_adapter.efficient_attention", new=nullcontext):
                raw = adapter._generate(targets.file_path.iloc[0], "fixed prompt", examples=examples)
            self.assertEqual(parse_prediction(raw)["severity"], "2")
            self.assertEqual(processor.image_modes, ["RGB"] * 5)
            self.assertEqual([m["role"] for m in processor.messages], ["user", "assistant"] * 4 + ["user"])
            self.assertIn("fixed prompt", processor.messages[0]["content"][1]["text"])
            self.assertEqual(processor.messages[-1]["content"][1]["text"],
                             "Target image: return only its JSON classification.")
            self.assertEqual(model.settings["max_new_tokens"], 128)
            self.assertFalse(model.settings["do_sample"])
            self.assertIsNone(adapter.inference_config["custom_pixel_cap"])

    def test_nonrecording_dispatch_does_not_retain_arguments(self):
        class Argument:
            pass
        argument = Argument()
        reference = weakref.ref(argument)
        self.assertFalse(disable_native_gqa(argument, key=argument))
        del argument
        gc.collect()
        self.assertIsNone(reference())

    def test_trm_recursive_gradients_detach_and_checkpoint_roundtrip(self):
        config = tiny_config()
        model = VisionTRM(config["model"], patch_count=1)
        pixels = torch.rand(3, 1, 3, 8, 8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        state = model.initial_state(3)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            state, logits = model.refine_once(model.encode(pixels), state)
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 3])).backward()
            self.assertFalse(state.answer.requires_grad)
            for component in (model.encoder, model.reasoner, model.classifier):
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in component.parameters()))
            optimizer.step()
        other = VisionTRM(config["model"], patch_count=1)
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(pixels), other(pixels))
        self.assertEqual(VisionTRM(json.loads((PROJECT / "configs/trm.json").read_text())["model"]).parameter_counts()["total"], 556836)

    def test_vendored_source_matches_provenance(self):
        provenance = json.loads(PROVENANCE_PATH.read_text())
        for record in provenance["files"]:
            actual = hashlib.sha256((PROVENANCE_PATH.parent / record["path"]).read_bytes()).hexdigest()
            self.assertEqual(actual, record["vendored_sha256"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime required")
    def test_cuda_efficient_attention(self):
        self.assertEqual(len(validate_efficient_attention()), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime required")
    def test_cuda_amp_training_excludes_test_pixels_and_loads_checkpoint(self):
        from src.trm_data import FaceSeverityDataset
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            frame = make_images(root / "dataset", people=4)
            config = tiny_config()
            accessed = []
            original = FaceSeverityDataset.__getitem__
            def tracked(dataset, index):
                accessed.append(dataset.frame.iloc[index].file_path)
                return original(dataset, index)
            with patch.object(FaceSeverityDataset, "__getitem__", tracked):
                path = train_experiment(frame, root / "dataset", root / "run", config, show_progress=False)
            model, checkpoint = load_checkpoint(path, "cuda")
            held_out = partition_frame(checkpoint["split"], "test", root / "dataset")
            self.assertFalse(set(accessed) & set(held_out.file_path))
            history = pd.read_csv(root / "run/training_history.csv")
            self.assertGreater(history.optimizer_updates.iloc[-1], 0)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                logits = model(torch.rand(2, 1, 3, 8, 8, device="cuda"))
            self.assertEqual(logits.dtype, torch.float16)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
