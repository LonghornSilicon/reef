"""Token embedding lookup table."""

import torch
from torch import nn


class Embedding(nn.Module):
    """Map integer ids to rows of a learned weight matrix."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weight[ids]
