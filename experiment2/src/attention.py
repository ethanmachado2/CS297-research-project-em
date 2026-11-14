"""T4-compatible efficient SDPA with explicit grouped-query KV expansion."""
from contextlib import contextmanager


def disable_native_gqa(*_arguments, **_keywords):
    # A plain callable is intentional: a recording mock retains CUDA tensors.
    return False


@contextmanager
def efficient_attention():
    import transformers.integrations.sdpa_attention as integration
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not hasattr(integration, "use_gqa_in_sdpa"):
        raise RuntimeError("Unsupported Transformers integration; install requirements.txt")
    original = integration.use_gqa_in_sdpa
    integration.use_gqa_in_sdpa = disable_native_gqa
    try:
        # No math fallback: unsupported kernels fail visibly rather than OOM.
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            yield
    finally:
        integration.use_gqa_in_sdpa = original


def validate_efficient_attention():
    """Small CUDA correctness check before loading the model."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    generator = torch.Generator(device="cuda:0").manual_seed(42)
    q = torch.randn(1, 8, 97, 256, device="cuda:0", dtype=torch.float16, generator=generator)
    k = torch.randn(1, 2, 97, 256, device="cuda:0", dtype=torch.float16, generator=generator)
    v = torch.randn(1, 2, 97, 256, device="cuda:0", dtype=torch.float16, generator=generator)
    checks = []
    with torch.inference_mode():
        for causal in (False, True):
            with sdpa_kernel(SDPBackend.MATH):
                expected = sdpa(q.float(), k.float(), v.float(), is_causal=causal, enable_gqa=True)
            with efficient_attention():
                actual = sdpa(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), is_causal=causal)
            torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=2e-3)
            checks.append({"causal": causal, "max_absolute_error":
                           (actual.float() - expected).abs().max().item()})
    return checks
