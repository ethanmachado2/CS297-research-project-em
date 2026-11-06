"""Continuous visual input and severity readout around the official TRM blocks."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn

from vendor.tiny_recursive_models.models.common import trunc_normal_init_
from vendor.tiny_recursive_models.models.layers import RotaryEmbedding
from vendor.tiny_recursive_models.models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Block as OfficialBlock,
    TinyRecursiveReasoningModel_ACTV1Config as OfficialConfig,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule as OfficialReasoner,
)

PROVENANCE_PATH = Path(__file__).resolve().parents[1] / "vendor/tiny_recursive_models/PROVENANCE.json"
UPSTREAM_SHA = json.loads(PROVENANCE_PATH.read_text())["commit"]


def validate_model_config(config):
    expected = {"hidden_size", "num_heads", "L_layers", "expansion", "H_cycles", "L_cycles", "refinement_steps"}
    if set(config) != expected:
        raise ValueError(f"Model configuration requires {sorted(expected)}")
    for key in expected - {"expansion"}:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not math.isfinite(config["expansion"]) or config["expansion"] <= 0:
        raise ValueError("expansion must be positive and finite")
    if config["hidden_size"] % config["num_heads"] or (config["hidden_size"] // config["num_heads"]) % 2:
        raise ValueError("Width must be divisible by heads with even rotary head dimension")


class PatchEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["hidden_size"]
        self.conv = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.projection = nn.Linear(64, width)

    def forward(self, patches):
        b, p, c, h, w = patches.shape
        return self.projection(self.conv(patches.reshape(b * p, c, h, w)).flatten(1)).reshape(b, p, -1)


@dataclass
class RecursiveState:
    answer: torch.Tensor
    working: torch.Tensor


class VisionTRM(nn.Module):
    def __init__(self, config, *, patch_count=9):
        super().__init__()
        validate_model_config(config)
        if type(patch_count) is not int or patch_count <= 0:
            raise ValueError("patch_count must be a positive integer")
        self.config, self.patch_count = dict(config), patch_count
        width = config["hidden_size"]
        official = OfficialConfig(batch_size=1, seq_len=patch_count + 1,
            num_puzzle_identifiers=0, vocab_size=4, puzzle_emb_ndim=0, puzzle_emb_len=0,
            H_layers=0, L_layers=config["L_layers"], H_cycles=config["H_cycles"], L_cycles=config["L_cycles"],
            hidden_size=width, num_heads=config["num_heads"], expansion=config["expansion"],
            pos_encodings="rope", halt_max_steps=config["refinement_steps"], halt_exploration_prob=0,
            forward_dtype="float32", mlp_t=False)
        self.encoder = PatchEncoder(config)
        self.readout = nn.Parameter(torch.empty(1, 1, width))
        trunc_normal_init_(self.readout, std=1 / math.sqrt(width))
        self.reasoner = OfficialReasoner([OfficialBlock(official) for _ in range(config["L_layers"])])
        self.rotary = RotaryEmbedding(width // config["num_heads"], patch_count + 1, 10000.0)
        self.register_buffer("answer_init", trunc_normal_init_(torch.empty(width), std=1))
        self.register_buffer("working_init", trunc_normal_init_(torch.empty(width), std=1))
        self.classifier = nn.Linear(width, 4)

    def encode(self, patches):
        if patches.ndim != 5 or patches.shape[1] != self.patch_count or patches.shape[2] != 3:
            raise ValueError(f"Expected [B,{self.patch_count},3,H,W] patches")
        image_vectors = self.encoder(patches)
        return torch.cat((self.readout.expand(patches.shape[0], -1, -1), image_vectors), dim=1) * math.sqrt(self.config["hidden_size"])

    def initial_state(self, batch_size, device=None):
        shape = (batch_size, self.patch_count + 1, self.config["hidden_size"])
        device = device or self.answer_init.device
        return RecursiveState(self.answer_init.to(device).expand(shape).clone(),
                              self.working_init.to(device).expand(shape).clone())

    def refine_once(self, encoded, state):
        answer, working = state.answer, state.working
        positional = self.rotary()
        def cycle(answer, working):
            for _ in range(self.config["L_cycles"]):
                working = self.reasoner(working, answer + encoded, cos_sin=positional)
            answer = self.reasoner(answer, working, cos_sin=positional)
            return answer, working
        with torch.no_grad():
            for _ in range(self.config["H_cycles"] - 1):
                answer, working = cycle(answer, working)
        answer, working = cycle(answer, working)
        logits = self.classifier(answer[:, 0])
        return RecursiveState(answer.detach(), working.detach()), logits

    def forward(self, patches):
        encoded = self.encode(patches)
        state = self.initial_state(patches.shape[0], patches.device)
        for _ in range(self.config["refinement_steps"]):
            state, logits = self.refine_once(encoded, state)
        return logits

    def parameter_counts(self):
        count = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {"encoder": count(self.encoder), "reasoning": count(self.reasoner),
                "classifier": count(self.classifier), "readout": self.readout.numel(), "total": count(self)}
