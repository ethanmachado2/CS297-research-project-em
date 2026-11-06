"""Read-only stage tracing and frozen-feature probes for the face TRM.

Diagnostic probes use training labels only; validation is scored separately.
They are localization tools, not replacement experiment results.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def summarize_features(features, targets):
    """Across-image variation and variance explained by class centroids."""
    x = features.detach().double().flatten(1)
    centered = x - x.mean(0)
    variance = centered.square().mean()
    between = sum((targets == label).float().mean().item()
                  * (x[targets == label].mean(0) - x.mean(0)).square().mean()
                  for label in targets.unique())
    rms = x.square().mean().sqrt()
    return {"dimensions": x.shape[1], "rms": rms.item(),
            "image_std_rms": variance.sqrt().item(),
            "relative_image_variation": (variance.sqrt() / rms.clamp_min(1e-30)).item(),
            "between_class_variance_fraction": (between / variance.clamp_min(1e-30)).item()}


def ridge_probe(train, targets, validation, validation_targets, alpha=1.0):
    """Fixed-alpha linear probe; scaling and fitting see training data only."""
    train, validation = train.double().flatten(1), validation.double().flatten(1)
    mean, scale = train.mean(0), train.std(0, correction=0).clamp_min(1e-8)
    x, v = (train - mean) / scale, (validation - mean) / scale
    labels = nn.functional.one_hot(targets, 4).double()
    prior = labels.mean(0)
    # Dual solution is cheaper for the many-dimensional pooled CNN features.
    weight = x.T @ torch.linalg.solve(x @ x.T + alpha * torch.eye(len(x), dtype=x.dtype), labels - prior)
    def score(z, y):
        guess = (z @ weight + prior).argmax(-1)
        confusion = torch.bincount(y * 4 + guess, minlength=16).reshape(4, 4)
        route_accuracy = ((guess >= 2) == (y >= 2)).double().mean().item()
        return {"severity_accuracy": (guess == y).double().mean().item(),
                "route_accuracy": route_accuracy, "confusion": confusion.tolist()}
    return {"alpha": alpha, "train": score(x, targets), "validation": score(v, validation_targets)}


def centroid_probe(train, targets, validation, validation_targets):
    """Nearest class centroid in train-standardized space; no tuned thresholds."""
    train, validation = train.double().flatten(1), validation.double().flatten(1)
    mean, scale = train.mean(0), train.std(0, correction=0).clamp_min(1e-8)
    x, v = (train - mean) / scale, (validation - mean) / scale
    centroids = torch.stack([x[targets == i].mean(0) for i in range(4)])
    def score(z, y):
        guess = torch.cdist(z, centroids).argmin(-1)
        return {"severity_accuracy": (guess == y).double().mean().item(),
                "route_accuracy": ((guess >= 2) == (y >= 2)).double().mean().item(),
                "confusion": torch.bincount(y * 4 + guess, minlength=16).reshape(4, 4).tolist()}
    return {"train": score(x, targets), "validation": score(v, validation_targets)}


def trace_features(model, patches):
    """Trace the actual forward path with temporary hooks, without mutation."""
    features, probabilities, handles = {}, [], []
    def record(name, tensor):
        features[name] = tensor.detach().cpu()
    batch, count = patches.shape[:2]
    # A simple diagnostic of local high-frequency energy, not a model input.
    gray = (patches * patches.new_tensor([.299, .587, .114])[None, None, :, None, None]).sum(2)
    residual = (gray[..., 1:, 1:] - gray[..., :-1, 1:]
                - gray[..., 1:, :-1] + gray[..., :-1, :-1]) / 2
    record("input/residual_std", residual.flatten(1).std(1, correction=0)[:, None])
    record("input/rgb_mean_std", torch.cat((patches.mean((1, 3, 4)), patches.std((1, 3, 4), correction=0)), -1))
    # Hooks run only when the real encoder executes below.
    for index, layer in enumerate(model.encoder.conv):
        if isinstance(layer, nn.GELU):
            stage = index // 2 + 1
            def capture(_module, _args, output, stage=stage):
                value = output.reshape(batch, count, *output.shape[1:])
                record(f"encoder/conv{stage}_mean", value.mean((1, 3, 4)))
                record(f"encoder/conv{stage}_std", value.std((3, 4), correction=0).mean(1))
            handles.append(layer.register_forward_hook(capture))
    calls = []
    def capture_reasoner(_module, _args, output):
        calls.append(output)
        index = len(calls)
        record(f"reasoner/{index:02d}_readout", output[:, 0])
        record(f"reasoner/{index:02d}_patch_mean", output[:, 1:].mean(1))
    handles.append(model.reasoner.register_forward_hook(capture_reasoner))
    try:
        with torch.inference_mode():
            encoded = model.encode(patches)
            record("encoder/projected_patch_mean", encoded[:, 1:].mean(1))
            record("encoder/projected_patch_flat", encoded[:, 1:].flatten(1))
            state = model.initial_state(batch, patches.device)
            for step in range(model.config["refinement_steps"]):
                state, logits = model.refine_once(encoded, state)
                record(f"outer/{step + 1}_answer", state.answer[:, 0])
                record(f"outer/{step + 1}_logits", logits)
                probabilities.append(logits.softmax(-1).cpu())
    finally:
        for handle in handles:
            handle.remove()
    return features, probabilities


def gradient_trace(model, patches, targets):
    """Measure each supervised step without updating checkpoint parameters."""
    model.zero_grad(set_to_none=True)
    state = model.initial_state(len(targets), patches.device)
    result = []
    for step in range(model.config["refinement_steps"]):
        model.zero_grad(set_to_none=True)
        encoded = model.encode(patches)
        encoded.retain_grad()
        state, logits = model.refine_once(encoded, state)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        groups = {}
        for name, component in (("encoder", model.encoder), ("reasoner", model.reasoner),
                                ("classifier", model.classifier)):
            parameters = list(component.parameters())
            norm = math.sqrt(sum(p.grad.double().square().sum().item() for p in parameters if p.grad is not None))
            parameter_norm = math.sqrt(sum(p.detach().double().square().sum().item() for p in parameters))
            groups[name] = {"gradient_norm": norm, "parameter_norm": parameter_norm,
                            "gradient_parameter_ratio": norm / max(parameter_norm, 1e-30),
                            "missing_gradients": sum(p.grad is None for p in parameters)}
        total_norm = math.sqrt(sum(p.grad.double().square().sum().item() for p in model.parameters() if p.grad is not None))
        result.append({"step": step + 1, "loss": loss.item(), "groups": groups,
                       "total_gradient_norm": total_norm,
                       "encoded_patch_gradient_norm": encoded.grad[:, 1:].double().norm().item(),
                       "encoded_readout_gradient_norm": encoded.grad[:, :1].double().norm().item()})
    model.zero_grad(set_to_none=True)
    return result


def compare_gradient_truncation(model, patches, targets):
    """Same one-step forward computation, with/without the inner no-grad cycle.

    This changes only autograd coverage, without an optimizer update. It does
    not test removal of the detached boundary between outer steps.
    """
    results, references = {}, []
    for full in (False, True):
        model.zero_grad(set_to_none=True)
        encoded = model.encode(patches)
        encoded.retain_grad()
        state = model.initial_state(len(targets), patches.device)
        answer, working = state.answer, state.working
        for cycle in range(model.config["H_cycles"]):
            with torch.set_grad_enabled(full or cycle == model.config["H_cycles"] - 1):
                for _ in range(model.config["L_cycles"]):
                    working = model.reasoner(working, answer + encoded, cos_sin=model.rotary())
                answer = model.reasoner(answer, working, cos_sin=model.rotary())
        logits = model.classifier(answer[:, 0])
        loss = nn.functional.cross_entropy(logits, targets)
        references.append(logits.detach())
        loss.backward()
        results["all_inner_cycles" if full else "default_truncated"] = {
            "loss": loss.item(), "encoder_gradient_norm": math.sqrt(sum(
                p.grad.double().square().sum().item() for p in model.encoder.parameters())),
            "encoded_patch_gradient_norm": encoded.grad[:, 1:].double().norm().item()}
    torch.testing.assert_close(*references, rtol=0, atol=0)
    model.zero_grad(set_to_none=True)
    return results
