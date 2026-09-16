"""Shared tolerances and helpers for reference-comparison tests."""

import torch

DTYPES = (torch.float32, torch.bfloat16)

# Worst measured bfloat16 deviation from the fused kernels is ~1.6e-2 absolute
# (convolution), far above assert_close's default atol of 1e-5.
BF16_TOLERANCE = {"rtol": 2e-2, "atol": 2e-2}


def assert_matches(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> None:
    tolerance = {} if dtype == torch.float32 else BF16_TOLERANCE
    torch.testing.assert_close(actual, expected, **tolerance)
