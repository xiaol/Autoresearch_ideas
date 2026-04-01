from __future__ import annotations

import torch
from torch import nn


class DeepEmbed(nn.Module):
    """Token-conditional channel modulation inspired by RWKV v8 DeepEmbed.

    This is a lightweight, per-layer modulation. It is not a full RWKV v8 implementation.
    """

    def __init__(self, vocab_size: int, d_model: int, deep_dim: int):
        super().__init__()
        self.deep = nn.Embedding(vocab_size, deep_dim)
        self.proj = nn.Linear(deep_dim, d_model, bias=False)

    def forward(self, token_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T), x: (B, T, D)
        mod = torch.sigmoid(self.proj(self.deep(token_ids)))
        return x * (1.0 + mod)
