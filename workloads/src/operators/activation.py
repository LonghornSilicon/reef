"""Elementwise activation functions built from primitive tensor ops."""

import torch
from torch import nn


class ReLU(nn.Module):
    """Rectified linear unit."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp negative entries of ``x`` to zero.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of the same shape with negatives replaced by zero.
        """
        return torch.where(x > 0, x, torch.zeros_like(x))


class SiLU(nn.Module):
    """Sigmoid linear unit, ``x * sigmoid(x)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SiLU activation to ``x``.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of the same shape holding ``x * sigmoid(x)``.
        """
        return x * (1.0 / (1.0 + torch.exp(-x)))


class Softmax(nn.Module):
    """Numerically stable softmax over a single axis."""

    def __init__(self, dim: int = -1) -> None:
        """Store the axis to normalize over.

        Args:
            dim: Axis along which the outputs sum to one.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` along ``self.dim``.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of the same shape summing to one along ``self.dim``.
        """
        shifted = x - x.amax(dim=self.dim, keepdim=True)
        exponentiated = torch.exp(shifted)
        return exponentiated / exponentiated.sum(dim=self.dim, keepdim=True)
