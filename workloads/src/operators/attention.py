"""Grouped-query scaled dot-product attention."""

import torch
from torch import nn

from operators.activation import Softmax


class GroupedQueryAttention(nn.Module):
    """Attention where several query heads share one key/value head."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float | None = None,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.repeats = num_heads // num_kv_heads
        self.scale = head_dim**-0.5 if scale is None else scale
        self.sliding_window = sliding_window
        self.softmax = Softmax(dim=-1)

    def expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.repeats == 1:
            return x
        batch, heads, length, dim = x.shape
        expanded = x[:, :, None].expand(batch, heads, self.repeats, length, dim)
        return expanded.reshape(batch, heads * self.repeats, length, dim)

    def causal_mask(
        self, query_len: int, key_len: int, device: torch.device
    ) -> torch.Tensor:
        """Return a ``(query_len, key_len)`` mask, true where disallowed."""
        # Queries occupy the final query_len positions of the key axis, so
        # cached decoding gets the right offset for free.
        query_pos = torch.arange(key_len - query_len, key_len, device=device)
        key_pos = torch.arange(key_len, device=device)
        relative = key_pos[None, :] - query_pos[:, None]
        masked = relative > 0
        if self.sliding_window is not None:
            # The window includes the query's own position.
            masked = masked | (relative <= -self.sliding_window)
        return masked

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # query: (batch, num_heads, query_len, head_dim)
        # key, value: (batch, num_kv_heads, key_len, head_dim)
        # mask: bool, true where disallowed; bias: additive scores. Both
        # broadcast to (batch, num_heads, query_len, key_len).
        key = self.expand_kv(key)
        value = self.expand_kv(value)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if bias is not None:
            # Add before the fill: finfo.min plus a negative bias overflows
            # to -inf and a fully masked row would softmax to NaN.
            scores = scores + bias
        if causal:
            disallowed = self.causal_mask(
                query.shape[2], key.shape[2], query.device
            )
            mask = disallowed if mask is None else disallowed | mask
        if mask is not None:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = self.softmax(scores.to(torch.float32)).to(value.dtype)
        return torch.matmul(weights, value)
