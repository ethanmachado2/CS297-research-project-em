"""Noise-aware ViTRM adaptation of arXiv:2603.19503v2, equations 2–3.

This is an independent implementation, not a reproduction of CIFAR results.
Local CNN/RGB moments replace linear patch embeddings. All refinements share
one Transformer, and gradients span the complete fixed-depth unroll.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

IMPLEMENTATION = "noise_aware_vitrm_v1"
PAPER_URL = "https://arxiv.org/abs/2603.19503v2"


def validate_model_config(config):
    expected = {"hidden_size", "num_heads", "layers", "expansion", "memory_tokens",
                "token_grid", "memory_updates", "refinement_steps"}
    if set(config) != expected:
        raise ValueError(f"ViTRM model configuration requires {sorted(expected)}")
    for key in expected - {"expansion"}:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["hidden_size"] % config["num_heads"]:
        raise ValueError("hidden_size must be divisible by num_heads")
    if not math.isfinite(config["expansion"]) or config["expansion"] <= 0:
        raise ValueError("expansion must be positive and finite")


class MomentEncoder(nn.Module):
    """Spatial feature moments plus RGB moments retain local noise amplitude.

    Moments are reduced in FP32 under AMP. No per-image variance normalization
    is applied. RGB moments provide a short path around CNN downsampling.
    """
    def __init__(self, width, grid):
        super().__init__()
        self.grid = grid
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU())
        self.projection = nn.Linear(2 * (64 + 3), width)

    def moments(self, value):
        value = value.float()
        mean = F.adaptive_avg_pool2d(value, self.grid)
        variance = (F.adaptive_avg_pool2d(value.square(), self.grid) - mean.square()).clamp_min(0)
        return torch.cat((mean, (variance + 1e-8).sqrt()), dim=1)

    def forward(self, patches):
        batch, crops, channels, height, width = patches.shape
        pixels = patches.reshape(batch * crops, channels, height, width)
        features = self.conv(pixels)
        if min(features.shape[-2:]) < self.grid:
            raise ValueError("token_grid exceeds the CNN feature map; increase patch_size or reduce token_grid")
        moments = torch.cat((self.moments(features), self.moments(pixels)), dim=1)
        tokens = moments.flatten(2).transpose(1, 2)
        return self.projection(tokens).reshape(batch, crops * self.grid ** 2, -1)


class VisionViTRM(nn.Module):
    def __init__(self, config, *, patch_count=9):
        super().__init__()
        validate_model_config(config)
        if type(patch_count) is not int or patch_count <= 0:
            raise ValueError("patch_count must be a positive integer")
        self.config, self.patch_count = dict(config), patch_count
        width, grid = config["hidden_size"], config["token_grid"]
        self.encoder = MomentEncoder(width, grid)
        self.crop_position = nn.Parameter(torch.empty(1, patch_count, 1, width))
        self.cell_position = nn.Parameter(torch.empty(1, 1, grid ** 2, width))
        self.answer_init = nn.Parameter(torch.empty(1, 1, width))
        self.memory_init = nn.Parameter(torch.empty(1, config["memory_tokens"], width))
        # One stack is reused for BOTH update phases and all refinements.
        self.reasoner = nn.Sequential(*[
            nn.TransformerEncoderLayer(width, config["num_heads"],
                dim_feedforward=max(1, int(width * config["expansion"])),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(config["layers"])])
        self.output_norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 4)
        for parameter in (self.crop_position, self.cell_position, self.answer_init, self.memory_init):
            nn.init.trunc_normal_(parameter, std=0.02)

    def encode(self, patches):
        if patches.ndim != 5 or patches.shape[1] != self.patch_count or patches.shape[2] != 3:
            raise ValueError(f"Expected [B,{self.patch_count},3,H,W] patches")
        tokens = self.encoder(patches)
        positions = (self.crop_position + self.cell_position).flatten(1, 2)
        return tokens + positions.to(tokens.dtype)

    def refine_tokens(self, image, answer, memory, step):
        """One shared-weight refinement; image tokens are read only."""
        for _ in range(self.config["memory_updates"]):
            memory = self.reasoner(torch.cat((image, answer, memory), dim=1))[:, -memory.shape[1]:]
        answer = self.reasoner(torch.cat((answer, memory), dim=1))[:, :1]
        return answer, memory

    def forward_tokens(self, image, *, steps=None):
        """Evaluate a prefix of the trained unroll from one encoded tensor.

        A prefix always starts with the learned initial states. No extrapolation
        beyond the configured training depth is allowed.
        """
        steps = self.config["refinement_steps"] if steps is None else steps
        if type(steps) is not int or not 1 <= steps <= self.config["refinement_steps"]:
            raise ValueError("steps must be within the configured training depth")
        expected_tokens = self.patch_count * self.config["token_grid"] ** 2
        if image.ndim != 3 or image.shape[1:] != (expected_tokens, self.config["hidden_size"]):
            raise ValueError("Unexpected encoded image token shape")
        answer = self.answer_init.to(image.dtype).expand(len(image), -1, -1)
        memory = self.memory_init.to(image.dtype).expand(len(image), -1, -1)
        outputs = []
        for step in range(steps):
            answer, memory = self.refine_tokens(image, answer, memory, step)
            outputs.append(self.classifier(self.output_norm(answer[:, 0])))
        return torch.stack(outputs, dim=1)

    def forward_steps(self, patches, *, steps=None):
        """Encode once and return [B,R,4] logits with fresh initial states."""
        return self.forward_tokens(self.encode(patches), steps=steps)

    def forward(self, patches):
        return self.forward_steps(patches)[:, -1]

    def parameter_counts(self):
        count = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {"encoder": count(self.encoder), "reasoning": count(self.reasoner),
                "classifier": count(self.classifier) + count(self.output_norm),
                "states_and_positions": sum(p.numel() for p in
                    (self.crop_position, self.cell_position, self.answer_init, self.memory_init)),
                "total": count(self)}
