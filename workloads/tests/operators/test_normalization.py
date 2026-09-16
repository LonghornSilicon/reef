"""RMSNorm against the reference ``Qwen3RMSNorm`` from transformers."""

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from operators.normalization import RMSNorm
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

SHAPES = [(4, 32), (2, 5, 32), (2, 4, 6, 32)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_rmsnorm_matches_reference(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    """RMSNorm matches ``Qwen3RMSNorm`` with the same learned gain."""
    ours = RMSNorm(32, eps=1e-6)
    ours.weight.data.normal_()
    reference = Qwen3RMSNorm(32, eps=1e-6)
    reference.weight.data.copy_(ours.weight.data)
    ours, reference = ours.to(dtype), reference.to(dtype)

    x = torch.randn(*shape, dtype=dtype)
    assert_matches(ours(x), reference(x), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_preserves_dtype(dtype: torch.dtype) -> None:
    """The float32 reduction does not leak into the output dtype."""
    layer = RMSNorm(32).to(dtype)
    assert layer(torch.randn(2, 32, dtype=dtype)).dtype == dtype
