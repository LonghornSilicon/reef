"""Linear against ``torch.nn.functional.linear``."""

import pytest
import torch
import torch.nn.functional as F

from tests.common import DTYPES, assert_matches
from workloads.operators.linear import Linear

pytestmark = pytest.mark.unit

SHAPES = [(4, 32), (2, 5, 32), (2, 3, 5, 32)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("shape", SHAPES)
def test_linear_matches_reference(
    shape: tuple[int, ...], bias: bool, dtype: torch.dtype
) -> None:
    """2-D to 4-D inputs, with and without bias, against ``F.linear``."""
    layer = Linear(32, 16, bias=bias).to(dtype)
    x = torch.randn(*shape, dtype=dtype)
    expected = F.linear(x, layer.weight, layer.bias)
    assert_matches(layer(x), expected, dtype)


def test_linear_without_bias_registers_none() -> None:
    """``bias=False`` leaves ``bias`` as None, as `torch.nn.Linear` does."""
    assert Linear(8, 4, bias=False).bias is None
