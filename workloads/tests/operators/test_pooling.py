"""Pooling operators against their ``torch.nn.functional`` equivalents."""

import pytest
import torch
import torch.nn.functional as F

from operators.pooling import AdaptiveAvgPool2d, MaxPool2d
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

# (input_size, kernel_size, stride)
MAXPOOL_CASES = [
    (13, 3, 2),
    (14, 3, 2),
    (16, 2, 2),
    (8, 3, 1),
    (55, 3, 2),  # AlexNet's first pooling stage at 224x224 input.
]

# (input_size, kernel_size, stride, padding)
MAXPOOL_PADDED_CASES = [
    (56, 3, 2, 1),  # ResNet's stem pooling at 224x224 input.
    (16, 3, 2, 1),
    (8, 3, 1, 1),
    (7, 2, 2, 1),  # Padding larger than the ragged edge it covers.
]

# (input_h, input_w, output_h, output_w)
ADAPTIVE_CASES = [
    (13, 13, 6, 6),  # Not divisible: windows overlap.
    (7, 7, 6, 6),  # AlexNet's avgpool at 256x256 input.
    (6, 6, 6, 6),  # Identity.
    (12, 12, 6, 6),  # Exactly divisible.
    (10, 10, 3, 4),  # Non-square output.
    (5, 9, 4, 2),  # Non-square input and output.
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("size", "kernel", "stride"), MAXPOOL_CASES)
def test_max_pool2d_matches_reference(
    size: int, kernel: int, stride: int, dtype: torch.dtype
) -> None:
    x = torch.randn(2, 3, size, size, dtype=dtype)
    pool = MaxPool2d(kernel_size=kernel, stride=stride)
    assert_matches(pool(x), F.max_pool2d(x, kernel, stride), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool2d_stride_defaults_to_kernel_size(
    dtype: torch.dtype,
) -> None:
    x = torch.randn(2, 3, 12, 12, dtype=dtype)
    pool = MaxPool2d(kernel_size=3)
    assert_matches(pool(x), F.max_pool2d(x, 3), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("size", "kernel", "stride", "padding"), MAXPOOL_PADDED_CASES
)
def test_max_pool2d_padding_matches_reference(
    size: int, kernel: int, stride: int, padding: int, dtype: torch.dtype
) -> None:
    x = torch.randn(2, 3, size, size, dtype=dtype)
    pool = MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)
    expected = F.max_pool2d(x, kernel, stride, padding)
    assert_matches(pool(x), expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool2d_padding_is_not_zero_fill(dtype: torch.dtype) -> None:
    x = -torch.rand(1, 1, 4, 4, dtype=dtype) - 1.0
    pool = MaxPool2d(kernel_size=3, stride=1, padding=1)
    assert (pool(x) < 0).all()
    assert_matches(pool(x), F.max_pool2d(x, 3, 1, 1), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("height", "width", "out_h", "out_w"), ADAPTIVE_CASES)
def test_adaptive_avg_pool2d_matches_reference(
    height: int, width: int, out_h: int, out_w: int, dtype: torch.dtype
) -> None:
    x = torch.randn(2, 3, height, width, dtype=dtype)
    pool = AdaptiveAvgPool2d((out_h, out_w))
    expected = F.adaptive_avg_pool2d(x, (out_h, out_w))
    assert_matches(pool(x), expected, dtype)
