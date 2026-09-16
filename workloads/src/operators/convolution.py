"""Two-dimensional convolution expressed as im2col plus a matmul."""

import math

import torch
from torch import nn


class Conv2d(nn.Module):
    """Square-kernel 2D convolution with uniform stride and zero padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        """Allocate the filter bank and optional bias.

        Args:
            in_channels: Channel count of the input feature map.
            out_channels: Number of filters, i.e. output channels.
            kernel_size: Height and width of each square filter.
            stride: Step between consecutive filter placements.
            padding: Zeros added on every side of the input.
            bias: Whether to add a learned per-filter bias.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1.0 / math.sqrt(fan_in)
        self.weight = nn.Parameter(
            torch.empty(
                out_channels, in_channels, kernel_size, kernel_size
            ).uniform_(-bound, bound)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_channels).uniform_(-bound, bound)
            )
        else:
            self.register_parameter("bias", None)

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """Surround ``x`` with ``self.padding`` rows and columns of zeros.

        Args:
            x: Tensor shaped ``(batch, channels, height, width)``.

        Returns:
            Zero-padded tensor, or ``x`` itself when padding is zero.
        """
        if self.padding == 0:
            return x
        batch, channels, height, width = x.shape
        size = (
            batch,
            channels,
            height + 2 * self.padding,
            width + 2 * self.padding,
        )
        padded = torch.zeros(size, dtype=x.dtype, device=x.device)
        row_low, row_high = self.padding, self.padding + height
        col_low, col_high = self.padding, self.padding + width
        padded[:, :, row_low:row_high, col_low:col_high] = x
        return padded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve ``x`` with the filter bank.

        Args:
            x: Tensor shaped ``(batch, in_channels, height, width)``.

        Returns:
            Tensor shaped ``(batch, out_channels, out_h, out_w)``.
        """
        padded = self.pad(x)
        # im2col via strided views: (batch, in_ch, out_h, out_w, k, k)
        patches = padded.unfold(2, self.kernel_size, self.stride).unfold(
            3, self.kernel_size, self.stride
        )
        batch, _, out_h, out_w = patches.shape[:4]
        columns = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            batch,
            out_h * out_w,
            self.in_channels * self.kernel_size * self.kernel_size,
        )
        flat_weight = self.weight.reshape(self.out_channels, -1).transpose(0, 1)
        out = torch.matmul(columns, flat_weight)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(batch, out_h, out_w, self.out_channels).permute(
            0, 3, 1, 2
        )
