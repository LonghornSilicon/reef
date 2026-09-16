"""Elementwise activation functions built from primitive tensor ops."""

import torch
from torch import nn


class ReLU(nn.Module):
    """Rectified linear unit."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, torch.zeros_like(x))


class SiLU(nn.Module):
    """Sigmoid linear unit, ``x * sigmoid(x)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch.sigmoid rather than 1 / (1 + exp(-x)): the explicit form
        # overflows exp in float16 once x drops below about -11, which flushes
        # the result to zero instead of the correct small negative value.
        return x * torch.sigmoid(x)


class Softmax(nn.Module):
    """Numerically stable softmax over a single axis."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted = x - x.amax(dim=self.dim, keepdim=True)
        exponentiated = torch.exp(shifted)
        return exponentiated / exponentiated.sum(dim=self.dim, keepdim=True)


class GELU(nn.Module):
    """Gaussian error linear unit, tanh approximation.

    This is the ``gelu_pytorch_tanh`` variant that Gemma 3 specifies, not the
    exact erf form; the two differ by ~1e-3 at the peak, which is well outside
    checkpoint-equivalence tolerances.
    """

    #: sqrt(2 / pi).
    COEFFICIENT = 0.7978845608028654

    CUBIC = 0.044715

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inner = self.COEFFICIENT * (x + self.CUBIC * x.pow(3))
        return 0.5 * x * (1.0 + torch.tanh(inner))
