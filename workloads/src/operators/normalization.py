"""Normalization operators.

:class:`RMSNorm` is the transformer-side normalizer; :class:`BatchNorm2d` is
the CNN-side one. They sit together because a hardware model cares about the
same thing in both: a reduction over one axis followed by an affine rescale.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Scale activations by their root mean square, then apply a gain."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Allocate the per-channel gain.

        Args:
            dim: Width of the trailing axis that is normalized.
            eps: Constant added to the variance for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the trailing axis of ``x``.

        The reduction runs in float32 regardless of the input dtype, matching
        the reference Qwen3 implementation.

        Args:
            x: Tensor shaped ``(..., dim)``.

        Returns:
            Tensor of the same shape and dtype as ``x``.
        """
        dtype = x.dtype
        promoted = x.to(torch.float32)
        variance = promoted.pow(2).mean(-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(dtype)


class BatchNorm2d(nn.Module):
    """Normalize each channel of a feature map by its batch statistics.

    Parameter and buffer names match ``torch.nn.BatchNorm2d`` so a torchvision
    ``state_dict`` loads without renaming: ``weight``, ``bias``,
    ``running_mean``, ``running_var`` and ``num_batches_tracked``.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        """Allocate the per-channel affine and the running statistics.

        Args:
            num_features: Number of channels, i.e. the size of axis 1.
            eps: Constant added to the variance for numerical stability.
            momentum: Weight given to the current batch when updating the
                running statistics.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )

    def _batch_statistics(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Measure this batch and fold the result into the running estimates.

        The normalization uses the biased variance while the running estimate
        takes the unbiased one, which is what ``torch.nn.BatchNorm2d`` does.

        Args:
            x: Tensor shaped ``(batch, channels, height, width)``.

        Returns:
            The batch mean and biased variance, each shaped ``(channels,)``.
        """
        reduced = (0, 2, 3)
        mean = x.mean(dim=reduced)
        variance = x.var(dim=reduced, correction=0)
        samples = x.numel() // x.shape[1]
        with torch.no_grad():
            self.num_batches_tracked += 1
            unbiased = variance * samples / (samples - 1)
            keep = 1.0 - self.momentum
            self.running_mean.mul_(keep).add_(self.momentum * mean)
            self.running_var.mul_(keep).add_(self.momentum * unbiased)
        return mean, variance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each channel, then apply the learned scale and shift.

        Args:
            x: Tensor shaped ``(batch, channels, height, width)``.

        Returns:
            Tensor of the same shape and dtype as ``x``.
        """
        if self.training:
            mean, variance = self._batch_statistics(x)
        else:
            mean, variance = self.running_mean, self.running_var
        shape = (1, -1, 1, 1)
        mean = mean.to(x.dtype).reshape(shape)
        variance = variance.to(x.dtype).reshape(shape)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        scaled = normalized * self.weight.reshape(shape)
        return scaled + self.bias.reshape(shape)


class GemmaRMSNorm(nn.Module):
    """RMS normalization in Gemma's parameterization.

    Three things differ from :class:`RMSNorm`, and all three change the
    numbers rather than just the spelling:

    * the gain is stored as an offset from one and applied as ``1 + weight``,
      so the parameter is zero-initialized rather than one-initialized;
    * the gain multiply happens in float32 and the result is cast afterwards,
      where :class:`RMSNorm` casts first and then multiplies;
    * the gain broadcasts against a trailing axis of ``dim``, which for the
      per-head Q/K norms is ``head_dim``.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Allocate the per-channel gain offset.

        Args:
            dim: Width of the trailing axis that is normalized.
            eps: Constant added to the variance for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the trailing axis of ``x``.

        Args:
            x: Tensor shaped ``(..., dim)``.

        Returns:
            Tensor of the same shape and dtype as ``x``.
        """
        promoted = x.to(torch.float32)
        variance = promoted.pow(2).mean(-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + self.eps)
        scaled = normalized * (1.0 + self.weight.to(torch.float32))
        return scaled.to(x.dtype)
