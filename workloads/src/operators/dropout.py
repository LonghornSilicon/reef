"""Inverted dropout regularization."""

import torch
from torch import nn


class Dropout(nn.Module):
    """Zero entries at random during training and rescale the survivors."""

    def __init__(self, p: float = 0.5) -> None:
        """Store the drop probability.

        Args:
            p: Probability that any given entry is zeroed.
        """
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Drop entries of ``x`` when training, otherwise pass it through.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of the same shape; ``x`` itself in eval mode.
        """
        if not self.training or self.p == 0.0:
            return x
        keep = (torch.rand_like(x) >= self.p).to(x.dtype)
        return x * keep / (1.0 - self.p)
