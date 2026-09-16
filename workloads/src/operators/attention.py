import torch
from torch import nn

from operators.activation import Softmax


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, num_heads: int, num_kv_heads: int, head_dim: int
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.repeats = num_heads // num_kv_heads
        self.scale = head_dim**-0.5
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
        key = self.expand_kv(key)
        value = self.expand_kv(value)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if causal:
            mask = self.causal_mask(query.shape[2], key.shape[2], query.device)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = self.softmax(scores.to(torch.float32)).to(value.dtype)
        return torch.matmul(weights, value)
