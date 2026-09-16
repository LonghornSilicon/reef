"""Normalization operators."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Scale activations by their root mean square, then apply a gain."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # float32 regardless of input dtype, matching the reference Qwen3.
        promoted = x.to(torch.float32)
        variance = promoted.pow(2).mean(-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(dtype)


class BatchNorm2d(nn.Module):
    """Normalize each channel of a feature map by its batch statistics."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
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
        reduced = (0, 2, 3)
        mean = x.mean(dim=reduced)
        variance = x.var(dim=reduced, correction=0)
        samples = x.numel() // x.shape[1]
        with torch.no_grad():
            self.num_batches_tracked += 1
            # Normalization uses the biased variance but the running estimate
            # takes the unbiased one, as torch.nn.BatchNorm2d does.
            unbiased = variance * samples / (samples - 1)
            keep = 1.0 - self.momentum
            self.running_mean.mul_(keep).add_(self.momentum * mean)
            self.running_var.mul_(keep).add_(self.momentum * unbiased)
        return mean, variance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """RMS normalization in Gemma's parameterization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        promoted = x.to(torch.float32)
        variance = promoted.pow(2).mean(-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + self.eps)
        # Multiply then cast, the reverse of RMSNorm, matching Gemma3RMSNorm.
        scaled = normalized * (1.0 + self.weight.to(torch.float32))
        return scaled.to(x.dtype)
