"""Conv2d against ``torch.nn.functional.conv2d``."""

import pytest
import torch
import torch.nn.functional as F

from operators.convolution import Conv2d
from tests.common import DTYPES, assert_matches

pytestmark = pytest.mark.unit

# (in_channels, out_channels, kernel_size, stride, padding, bias)
CASES = [
    (3, 8, 3, 1, 0, True),
    (3, 8, 3, 1, 1, True),
    (6, 4, 5, 2, 2, True),
    (4, 4, 1, 1, 0, False),
    (2, 5, 3, 2, 0, False),
    (3, 64, 11, 4, 2, True),  # AlexNet's first convolution.
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("in_channels", "out_channels", "kernel", "stride", "padding", "bias"),
    CASES,
)
def test_conv2d_matches_reference(
    in_channels: int,
    out_channels: int,
    kernel: int,
    stride: int,
    padding: int,
    bias: bool,
    dtype: torch.dtype,
) -> None:
    conv = Conv2d(
        in_channels,
        out_channels,
        kernel,
        stride=stride,
        padding=padding,
        bias=bias,
    ).to(dtype)
    x = torch.randn(2, in_channels, 17, 19, dtype=dtype)
    expected = F.conv2d(
        x, conv.weight, conv.bias, stride=stride, padding=padding
    )
    assert_matches(conv(x), expected, dtype)


def test_conv2d_pad_places_the_input_in_the_centre() -> None:
    conv = Conv2d(2, 2, 3, padding=2)
    x = torch.randn(1, 2, 4, 6)
    padded = conv.pad(x)
    assert padded.shape == (1, 2, 8, 10)
    torch.testing.assert_close(padded[:, :, 2:6, 2:8], x)
    assert padded.sum() == pytest.approx(x.sum().item(), rel=1e-6)
