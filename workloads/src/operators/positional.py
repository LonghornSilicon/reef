"""Rotary position embeddings."""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Precompute and apply rotary position embeddings."""

    def __init__(
        self,
        head_dim: int,
        theta: float = 10000.0,
        scaling_factor: float = 1.0,
        rotary_dim: int | None = None,
        inv_freq: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = head_dim if rotary_dim is None else rotary_dim
        self.scaling_factor = scaling_factor
        if inv_freq is None:
            exponent = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
            # Linear scaling divides positions; dividing inv_freq is
            # equivalent.
            inv_freq = 1.0 / (theta ** (exponent / self.rotary_dim))
            inv_freq = inv_freq / scaling_factor
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin tables, each ``(length, rotary_dim)``."""
        angles = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        # Split-half layout: both halves share the same frequency ordering.
        doubled = torch.cat((angles, angles), dim=-1)
        return doubled.cos().to(dtype), doubled.sin().to(dtype)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        rotated = x[..., : self.rotary_dim]
        rotated = rotated * cos + self.rotate_half(rotated) * sin
        if self.rotary_dim == self.head_dim:
            return rotated
        # Pythia and Phi rotate only the leading features.
        return torch.cat((rotated, x[..., self.rotary_dim :]), dim=-1)
