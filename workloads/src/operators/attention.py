"""Grouped-query scaled dot-product attention."""

import torch
from torch import nn

from operators.activation import Softmax


class GroupedQueryAttention(nn.Module):
    """Attention where several query heads share one key/value head."""

    def __init__(
        self, num_heads: int, num_kv_heads: int, head_dim: int
    ) -> None:
        """Store the head geometry and derived scale factor.

        Args:
            num_heads: Number of query heads.
            num_kv_heads: Number of key/value heads; must divide ``num_heads``.
            head_dim: Width of each head.
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.repeats = num_heads // num_kv_heads
        self.scale = head_dim**-0.5
        self.softmax = Softmax(dim=-1)

    def expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat each key/value head to cover its group of query heads.

        Args:
            x: Tensor shaped ``(batch, num_kv_heads, length, head_dim)``.

        Returns:
            Tensor shaped ``(batch, num_heads, length, head_dim)``.
        """
        if self.repeats == 1:
            return x
        batch, heads, length, dim = x.shape
        expanded = x[:, :, None].expand(batch, heads, self.repeats, length, dim)
        return expanded.reshape(batch, heads * self.repeats, length, dim)

    def causal_mask(
        self, query_len: int, key_len: int, device: torch.device
    ) -> torch.Tensor:
        """Build the boolean mask of disallowed query/key pairs.

        Args:
            query_len: Number of query positions.
            key_len: Number of key positions, including any cached ones.
            device: Device to allocate the mask on.

        Returns:
            Bool tensor shaped ``(query_len, key_len)``, true where masked.
        """
        # Queries occupy the final query_len positions of the key axis, so
        # cached decoding gets the right offset for free.
        query_pos = torch.arange(key_len - query_len, key_len, device=device)[
            :, None
        ]
        key_pos = torch.arange(key_len, device=device)[None, :]
        return key_pos > query_pos

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Attend ``query`` over ``key``/``value``.

        Args:
            query: Tensor shaped ``(batch, num_heads, query_len, head_dim)``.
            key: Tensor shaped ``(batch, num_kv_heads, key_len, head_dim)``.
            value: Tensor shaped ``(batch, num_kv_heads, key_len, head_dim)``.
            causal: Whether to forbid attending to later positions.

        Returns:
            Tensor shaped ``(batch, num_heads, query_len, head_dim)``.
        """
        key = self.expand_kv(key)
        value = self.expand_kv(value)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if causal:
            mask = self.causal_mask(query.shape[2], key.shape[2], query.device)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = self.softmax(scores.to(torch.float32)).to(value.dtype)
        return torch.matmul(weights, value)
