"""Two-dimensional convolution expressed as im2col plus a matmul."""

import math

import torch
from torch import nn

Padding = int | tuple[int, int] | tuple[int, int, int, int]


def expand_padding(padding: Padding) -> tuple[int, int, int, int]:
    """Return ``(left, right, top, bottom)``."""
    if isinstance(padding, int):
        return (padding, padding, padding, padding)
    if len(padding) == 2:
        height, width = padding
        return (width, width, height, height)
    return padding


def pad2d(
    x: torch.Tensor, padding: Padding, value: float = 0.0
) -> torch.Tensor:
    left, right, top, bottom = expand_padding(padding)
    if not (left or right or top or bottom):
        return x
    batch, channels, height, width = x.shape
    size = (batch, channels, height + top + bottom, width + left + right)
    padded = torch.full(size, value, dtype=x.dtype, device=x.device)
    padded[:, :, top : top + height, left : left + width] = x
    return padded


def windows2d(
    x: torch.Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
) -> torch.Tensor:
    """View ``(B, C, H, W)`` as ``(B, C, out_h, out_w, k_h, k_w)``, no copy."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return x.unfold(2, kernel_size[0], stride[0]).unfold(
        3, kernel_size[1], stride[1]
    )


def correlate(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    dilation: int,
    groups: int,
) -> torch.Tensor:
    """Cross-correlate an already padded ``x`` with ``weight``."""
    batch, in_channels = x.shape[:2]
    out_channels, _, kernel_size, _ = weight.shape
    window = dilation * (kernel_size - 1) + 1
    # Folding the group into the batch axis keeps the im2col view at rank 6.
    grouped = x.reshape(batch * groups, in_channels // groups, *x.shape[2:])
    # Tensor.unfold has no dilation, so take the dilated taps out of the full
    # window with a strided slice.
    patches = windows2d(grouped, window, stride)
    if dilation > 1:
        patches = patches[..., ::dilation, ::dilation]
    out_h, out_w = patches.shape[2:4]
    columns = patches.permute(0, 2, 3, 1, 4, 5).reshape(
        batch, groups, out_h * out_w, -1
    )
    flat_weight = weight.reshape(groups, out_channels // groups, -1)
    out = torch.matmul(columns, flat_weight.transpose(1, 2))
    return out.permute(0, 1, 3, 2).reshape(batch, out_channels, out_h, out_w)


def correlate_depthwise(
    x: torch.Tensor, weight: torch.Tensor, stride: int, dilation: int
) -> torch.Tensor:
    """Per-channel correlation as K² shifted multiply-adds, no im2col copy."""
    channels, _, kernel_size, _ = weight.shape
    height, width = x.shape[2:]
    window = dilation * (kernel_size - 1) + 1
    out_h = (height - window) // stride + 1
    out_w = (width - window) // stride + 1
    # Accumulate in float32: bf16 loses ~1e-2 over a 7x7 sum, the matmul path
    # accumulates in float32 too.
    out = torch.zeros(
        x.shape[0], channels, out_h, out_w, dtype=torch.float32, device=x.device
    )
    for u in range(kernel_size):
        for v in range(kernel_size):
            top, left = u * dilation, v * dilation
            tap = x[
                :,
                :,
                top : top + stride * (out_h - 1) + 1 : stride,
                left : left + stride * (out_w - 1) + 1 : stride,
            ]
            gain = weight[:, 0, u, v].reshape(1, -1, 1, 1)
            out = out + tap.to(torch.float32) * gain.to(torch.float32)
    return out.to(x.dtype)


class Conv2d(nn.Module):
    """Square-kernel 2D convolution with groups, dilation and zero padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Padding = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        fan_in = in_channels // groups * kernel_size * kernel_size
        bound = 1.0 / math.sqrt(fan_in)
        self.weight = nn.Parameter(
            torch.empty(
                out_channels, in_channels // groups, kernel_size, kernel_size
            ).uniform_(-bound, bound)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_channels).uniform_(-bound, bound)
            )
        else:
            self.register_parameter("bias", None)

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        return pad2d(x, self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = self.pad(x)
        if self.groups == self.in_channels == self.out_channels:
            out = correlate_depthwise(
                padded, self.weight, self.stride, self.dilation
            )
        else:
            out = correlate(
                padded, self.weight, self.stride, self.dilation, self.groups
            )
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        return out
