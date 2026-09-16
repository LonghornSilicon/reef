"""Activation operators against their torch equivalents."""

import pytest
import torch
import torch.nn.functional as F

from operators.activation import ReLU, SiLU, Softmax
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [-1, 0, 1, 2])
def test_softmax_matches_reference(dim: int, dtype: torch.dtype) -> None:
    """Softmax matches ``torch.softmax`` along every axis."""
    x = torch.randn(3, 5, 7, dtype=dtype)
    assert_matches(Softmax(dim=dim)(x), torch.softmax(x, dim=dim), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_is_stable_for_large_inputs(dtype: torch.dtype) -> None:
    """Softmax stays finite when the logits are far from zero."""
    x = torch.randn(4, 9, dtype=dtype) * 50.0
    actual = Softmax(dim=-1)(x)
    assert torch.isfinite(actual).all()
    assert_matches(actual, torch.softmax(x, dim=-1), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_matches_reference(dtype: torch.dtype) -> None:
    """SiLU matches ``F.silu``."""
    x = torch.randn(4, 6, 8, dtype=dtype)
    assert_matches(SiLU()(x), F.silu(x), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_relu_matches_reference(dtype: torch.dtype) -> None:
    """ReLU matches ``F.relu``."""
    x = torch.randn(4, 6, 8, dtype=dtype)
    assert_matches(ReLU()(x), F.relu(x), dtype)
