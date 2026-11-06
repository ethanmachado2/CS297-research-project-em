import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from src.trm_model import VisionTRM, OfficialBlock, OfficialReasoner, PROVENANCE_PATH
from trm_fixtures import tiny_config, PROJECT


class TRMModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.config = tiny_config()["model"]
        self.model = VisionTRM(self.config, patch_count=1)
        self.patches = torch.rand(3, 1, 3, 8, 8)

    def test_shapes_finiteness_and_parameter_count(self):
        for batch in (1, 3):
            patches = self.patches[:batch]
            self.assertEqual(self.model.encoder(patches).shape, (batch, 1, 16))
            logits = self.model(patches)
            self.assertEqual(logits.shape, (batch, 4))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertEqual(logits.dtype, torch.float32)
        config = json.loads((PROJECT / "configs/trm_experiment1.json").read_text())
        counts = VisionTRM(config["model"]).parameter_counts()
        self.assertEqual(counts["total"], 556836)
        self.assertEqual(counts["total"], sum(v for k, v in counts.items() if k != "total"))

    def test_two_supervised_updates_reach_all_components_and_detach(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        state = self.model.initial_state(3)
        for _ in range(2):
            before = {key: value.detach().clone() for key, value in self.model.named_parameters()}
            optimizer.zero_grad(set_to_none=True)
            encoded = self.model.encode(self.patches)
            encoded.retain_grad()
            state, logits = self.model.refine_once(encoded, state)
            nn.functional.cross_entropy(logits, torch.tensor([0, 1, 3])).backward()
            self.assertFalse(state.answer.requires_grad)
            self.assertFalse(state.working.requires_grad)
            self.assertGreater(encoded.grad.abs().sum().item(), 0)
            for component in (self.model.encoder, self.model.reasoner, self.model.classifier):
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters()))
                self.assertGreater(sum(p.grad.abs().sum().item() for p in component.parameters()), 0)
            optimizer.step()
            for prefix in ("encoder.", "reasoner.", "classifier."):
                self.assertTrue(any(not torch.equal(before[key], value) for key, value in self.model.named_parameters() if key.startswith(prefix)))

    def test_shared_recursion_budget_and_truncated_gradients(self):
        config = dict(self.config, H_cycles=2, L_cycles=2, refinement_steps=4)
        model = VisionTRM(config, patch_count=1)
        reason_calls, block_calls, gradient_modes = [], [], []
        handles = [model.reasoner.register_forward_hook(lambda *args: reason_calls.append(1))]
        handles.append(model.reasoner.register_forward_pre_hook(lambda *args: gradient_modes.append(torch.is_grad_enabled())))
        handles += [block.register_forward_hook(lambda *args: block_calls.append(1)) for block in model.reasoner.layers]
        try:
            model(self.patches)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(reason_calls), 24)
        self.assertEqual(len(block_calls), 48)
        self.assertEqual(gradient_modes, ([False] * 3 + [True] * 3) * 4)

    def test_new_image_resets_states(self):
        self.model.eval()
        with torch.inference_mode():
            first = self.model(self.patches[:1])
            self.model(self.patches[1:2])
            second = self.model(self.patches[:1])
        torch.testing.assert_close(first, second)

    def test_state_roundtrip(self):
        other = VisionTRM(self.config, patch_count=1)
        other.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(self.model(self.patches), other(self.patches))

    def test_invalid_dimensions_and_recursion(self):
        for key, value in (("H_cycles", 0), ("L_cycles", -1), ("refinement_steps", 0),
                           ("L_layers", 0), ("hidden_size", 15), ("num_heads", 16), ("expansion", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                VisionTRM(dict(self.config, **{key: value}))
        with self.assertRaises(ValueError):
            self.model(torch.rand(1, 2, 3, 8, 8))

    def test_official_classes_provenance_and_no_unused_heads(self):
        self.assertIsInstance(self.model.reasoner, OfficialReasoner)
        self.assertTrue(all(isinstance(block, OfficialBlock) for block in self.model.reasoner.layers))
        names = [name for name, _ in self.model.named_parameters()]
        self.assertFalse(any(word in name for name in names for word in ("puzzle_emb", "embed_tokens", "lm_head", "q_head")))
        provenance = json.loads(PROVENANCE_PATH.read_text())
        self.assertRegex(provenance["commit"], r"^[a-f0-9]{40}$")
        self.assertIn("MIT", (PROVENANCE_PATH.parent / "LICENSE").read_text())
        for record in provenance["files"]:
            self.assertEqual(hashlib.sha256((PROVENANCE_PATH.parent / record["path"]).read_bytes()).hexdigest(), record["vendored_sha256"])

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable in this process")
    def test_mps_forward_backward(self):
        model = self.model.to("mps")
        model(self.patches.to("mps")).sum().backward()
        self.assertEqual(model.answer_init.device.type, "mps")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_backward(self):
        model = self.model.to("cuda")
        model(self.patches.to("cuda")).sum().backward()
        self.assertEqual(model.answer_init.device.type, "cuda")
