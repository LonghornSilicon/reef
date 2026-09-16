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
    ) -> None:
        """Derive the inverse frequency table.

        Args:
            head_dim: Width of each attention head; must be even.
            theta: Base of the geometric frequency progression.
            scaling_factor: Linear context-extension factor. Dividing the
                frequencies by it is equivalent to dividing the positions by
                it, since the angles are their product; ``1.0`` disables it.
        """
        super().__init__()
        self.head_dim = head_dim
        self.scaling_factor = scaling_factor
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        self.register_buffer(
            "inv_freq",
            1.0 / (theta**exponent) / scaling_factor,
            persistent=False,
        )

    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the cosine and sine tables for ``positions``.

        Args:
            positions: Integer tensor shaped ``(length,)``.
            dtype: Dtype the tables are cast to.

        Returns:
            Pair of tensors each shaped ``(length, head_dim)``.
        """
        angles = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        # Split-half layout: both halves share the same frequency ordering.
        doubled = torch.cat((angles, angles), dim=-1)
        return doubled.cos().to(dtype), doubled.sin().to(dtype)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the two halves of the trailing axis into each other.

        Args:
            x: Tensor shaped ``(..., head_dim)``.

        Returns:
            Tensor of the same shape holding ``(-x_high, x_low)``.
        """
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Rotate ``x`` by the angles encoded in ``cos`` and ``sin``.

        Args:
            x: Tensor shaped ``(batch, heads, length, head_dim)``.
            cos: Cosine table shaped ``(length, head_dim)``.
            sin: Sine table shaped ``(length, head_dim)``.

        Returns:
            Tensor of the same shape as ``x``.
        """
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return x * cos + self.rotate_half(x) * sin
