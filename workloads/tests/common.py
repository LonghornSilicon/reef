"""Shared tolerances and helpers for reference-comparison tests."""

import torch

DTYPES = (torch.float32, torch.bfloat16)

# bfloat16 keeps 8 mantissa bits, and our operators accumulate in a different
# order than the fused PyTorch kernels they are checked against, so the
# default assert_close bounds for bfloat16 (rtol=1.6e-2, atol=1e-5) are far
# too tight on the absolute term. The worst deviation measured across this
# suite is ~1.6e-2 absolute, on convolution.
BF16_TOLERANCE = {"rtol": 2e-2, "atol": 2e-2}


def assert_matches(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> None:
    """Assert ``actual`` matches ``expected`` at a dtype-appropriate bound.

    float32 uses ``torch.testing.assert_close`` defaults; bfloat16 uses the
    looser :data:`BF16_TOLERANCE`.

    Args:
        actual: Output of the from-scratch operator.
        expected: Output of the PyTorch reference.
        dtype: Dtype both tensors were computed in.
    """
    tolerance = {} if dtype == torch.float32 else BF16_TOLERANCE
    torch.testing.assert_close(actual, expected, **tolerance)
