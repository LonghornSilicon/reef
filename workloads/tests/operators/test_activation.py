"""Activation operators against their torch equivalents."""

import pytest
import torch
import torch.nn.functional as F

from operators.activation import GELU, ReLU, SiLU, Softmax
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


@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu_matches_reference(dtype: torch.dtype) -> None:
    """GELU matches the ``tanh`` approximation PyTorch exposes."""
    x = torch.randn(4, 16, dtype=dtype)
    expected = F.gelu(x, approximate="tanh")
    assert_matches(GELU()(x), expected, dtype)


def test_gelu_uses_the_tanh_approximation_not_erf() -> None:
    """The tanh and erf forms differ; Gemma 3 specifies the tanh one."""
    x = torch.linspace(-4.0, 4.0, 200)
    ours = GELU()(x)
    torch.testing.assert_close(ours, F.gelu(x, approximate="tanh"))
    assert not torch.allclose(ours, F.gelu(x, approximate="none"), atol=1e-5)
