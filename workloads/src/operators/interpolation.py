"""Resampling operators built from per-axis index and weight tables."""

import math

import torch
from torch import nn

#: Keys' cubic convolution parameter, as torch's bicubic uses.
CUBIC_A = -0.75


def cubic_weights(t: torch.Tensor) -> torch.Tensor:
    """Return ``(n, 4)`` weights for the taps at offsets -1, 0, 1, 2."""
    a = CUBIC_A

    def near(x: torch.Tensor) -> torch.Tensor:
        return ((a + 2) * x - (a + 3)) * x * x + 1

    def far(x: torch.Tensor) -> torch.Tensor:
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a

    return torch.stack((far(t + 1), near(t), near(1 - t), far(2 - t)), dim=1)


def source_coordinates(
    out_size: int, in_size: int, ratio: float, align_corners: bool
) -> torch.Tensor:
    dst = torch.arange(out_size, dtype=torch.float32)
    if align_corners:
        scale = (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
        return dst * scale
    return (dst + 0.5) * ratio - 0.5


def resample_matrix(
    out_size: int, in_size: int, ratio: float, mode: str, align_corners: bool
) -> torch.Tensor:
    """Return ``(out_size, in_size)`` weights that resample one axis."""
    rows = torch.arange(out_size)
    if mode == "nearest":
        # Nearest uses floor(dst * ratio) with no half-pixel shift.
        taps = (rows.float() * ratio).floor().long().clamp(max=in_size - 1)
        taps = taps[:, None]
        weights = torch.ones(out_size, 1)
    elif mode == "bilinear":
        src = source_coordinates(out_size, in_size, ratio, align_corners)
        src = src.clamp(min=0.0)
        low = src.floor().long().clamp(max=in_size - 1)
        high = (low + 1).clamp(max=in_size - 1)
        frac = src - low
        taps = torch.stack((low, high), dim=1)
        weights = torch.stack((1 - frac, frac), dim=1)
    elif mode == "bicubic":
        src = source_coordinates(out_size, in_size, ratio, align_corners)
        low = src.floor()
        offsets = torch.arange(-1, 3)
        taps = (low.long()[:, None] + offsets[None, :]).clamp(0, in_size - 1)
        weights = cubic_weights(src - low)
    else:
        raise ValueError(f"unsupported mode {mode!r}")
    matrix = torch.zeros(out_size, in_size)
    matrix.index_put_(
        (rows[:, None].expand_as(taps), taps), weights, accumulate=True
    )
    return matrix


def gather_corners(
    flat: torch.Tensor, corners: list[tuple[torch.Tensor, torch.Tensor]]
) -> torch.Tensor:
    """Weighted sum of ``(N, P)`` taps from ``flat (C, L)``: ``(N, C, P)``."""
    channels = flat.shape[0]
    out = None
    for index, weight in corners:
        gathered = flat[:, index.long().reshape(-1)].reshape(
            channels, *index.shape
        )
        term = gathered.permute(1, 0, 2) * weight.to(flat.dtype)[:, None, :]
        out = term if out is None else out + term
    return out


class Interpolate(nn.Module):
    """Resize the last two axes by nearest, bilinear or bicubic sampling."""

    def __init__(
        self,
        size: int | tuple[int, int] | None = None,
        scale_factor: float | tuple[float, float] | None = None,
        mode: str = "nearest",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if (size is None) == (scale_factor is None):
            raise ValueError("give exactly one of size and scale_factor")
        self.size = (size, size) if isinstance(size, int) else size
        if isinstance(scale_factor, int | float):
            scale_factor = (scale_factor, scale_factor)
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def axis(self, in_size: int, index: int) -> tuple[int, float]:
        """Return the output size and the input/output ratio for one axis."""
        if self.size is not None:
            out_size = self.size[index]
            return out_size, in_size / out_size
        # With a scale factor torch measures the ratio from the factor, not
        # from the rounded output size.
        factor = self.scale_factor[index]
        return math.floor(in_size * factor), 1.0 / factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        out_h, ratio_h = self.axis(height, 0)
        out_w, ratio_w = self.axis(width, 1)
        rows = resample_matrix(
            out_h, height, ratio_h, self.mode, self.align_corners
        )
        cols = resample_matrix(
            out_w, width, ratio_w, self.mode, self.align_corners
        )
        rows = rows.to(dtype=x.dtype, device=x.device)
        cols = cols.to(dtype=x.dtype, device=x.device)
        return torch.matmul(rows, torch.matmul(x, cols.transpose(0, 1)))


class GridSample(nn.Module):
    """Bilinear sampling at normalized coordinates, zero outside the input."""

    def __init__(self, align_corners: bool = False) -> None:
        super().__init__()
        self.align_corners = align_corners

    def unnormalize(self, coords: torch.Tensor, size: int) -> torch.Tensor:
        if self.align_corners:
            return (coords + 1) / 2 * (size - 1)
        return ((coords + 1) * size - 1) / 2

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, height, width)
        # grid: (batch, out_h, out_w, 2) holding (x, y) in [-1, 1]
        batch, channels, height, width = x.shape
        out_h, out_w = grid.shape[1:3]
        # Coordinates in float32: F.grid_sample's bf16 kernel keeps them in
        # bf16 and picks the wrong tap near integers (measured 6e-2 off fp32).
        coords = grid.reshape(batch, -1, 2).to(torch.float32)
        ix = self.unnormalize(coords[..., 0], width)
        iy = self.unnormalize(coords[..., 1], height)
        x0, y0 = ix.floor(), iy.floor()
        x1, y1 = x0 + 1, y0 + 1
        wx1, wy1 = ix - x0, iy - y0
        wx0, wy0 = 1 - wx1, 1 - wy1
        flat = x.permute(1, 0, 2, 3).reshape(channels, -1)
        base = torch.arange(batch, device=x.device)[:, None] * (height * width)
        corners = []
        for xs, ys, weight in (
            (x0, y0, wx0 * wy0),
            (x1, y0, wx1 * wy0),
            (x0, y1, wx0 * wy1),
            (x1, y1, wx1 * wy1),
        ):
            inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            index = ys.clamp(0, height - 1) * width + xs.clamp(0, width - 1)
            corners.append((base + index, weight * inside))
        out = gather_corners(flat, corners)
        return out.reshape(batch, channels, out_h, out_w)
