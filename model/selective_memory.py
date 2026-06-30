from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class SelectiveMemoryConfig:
    d_model: int = 256
    n_slots: int = 32          # number of memory slots
    slot_dim: int = 64         # dimension of each slot key/value
    top_k: int = 8             # number of slots to read from
    write_gate_hidden: int = 64
    dropout: float = 0.1


class SelectiveMemory(nn.Module):
    """Selective online memory with delta-rule write gating.

    Maintains N key-value memory slots. On each forward pass:
    1. Compute write gate from input
    2. Compute key, query, value, and delta from input
    3. Delta-write to selected slots
    4. Read by attending to slots
    """

    def __init__(self, config: SelectiveMemoryConfig):
        super().__init__()
        self.config = config
        # Memory slots: learnable initial keys and values
        self.slot_keys = nn.Parameter(torch.randn(config.n_slots, config.slot_dim) * 0.02)
        self.slot_values = nn.Parameter(torch.randn(config.n_slots, config.slot_dim) * 0.02)

        # Projections
        self.key_proj = nn.Linear(config.d_model, config.slot_dim)
        self.query_proj = nn.Linear(config.d_model, config.slot_dim)
        self.value_proj = nn.Linear(config.d_model, config.slot_dim)
        self.delta_proj = nn.Linear(config.d_model, config.slot_dim)

        # Write gate
        self.write_gate = nn.Sequential(
            nn.Linear(config.d_model, config.write_gate_hidden),
            nn.GELU(),
            nn.Linear(config.write_gate_hidden, config.n_slots),
            nn.Sigmoid(),
        )

        # Output projection
        self.output_proj = nn.Linear(config.slot_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, D = x.shape
        device = x.device

        # Expand slots to batch
        keys = self.slot_keys.unsqueeze(0).expand(B, -1, -1)    # (B, N, slot_dim)
        vals = self.slot_values.unsqueeze(0).expand(B, -1, -1)  # (B, N, slot_dim)

        outputs = []
        for t in range(T):
            xt = x[:, t:t+1, :]  # (B, 1, D)

            # Write gate: how much to write to each slot
            gate = self.write_gate(xt.squeeze(1))  # (B, N)

            # Compute key, query, value, delta for this step
            k_t = self.key_proj(xt)      # (B, 1, slot_dim)
            q_t = self.query_proj(xt)    # (B, 1, slot_dim)
            v_t = self.value_proj(xt)    # (B, 1, slot_dim)
            d_t = self.delta_proj(xt)    # (B, 1, slot_dim)

            # Delta-rule write: keys update via delta, values update via delta
            # gate[:, :, None] is (B, N, 1), broadcast against (B, N, slot_dim)
            keys = keys + gate.unsqueeze(-1) * (k_t - keys) * 0.1
            vals = vals + gate.unsqueeze(-1) * (d_t - vals) * 0.1

            # Read: attend to slots using query
            attn_logits = torch.matmul(q_t, keys.transpose(-2, -1)) / (self.config.slot_dim ** 0.5)
            # (B, 1, N)

            # Top-k masking for sparsity
            if self.config.top_k < self.config.n_slots:
                topk_vals, _ = torch.topk(attn_logits, self.config.top_k, dim=-1)
                threshold = topk_vals[:, :, -1:]  # (B, 1, 1)
                attn_logits = torch.where(
                    attn_logits >= threshold,
                    attn_logits,
                    torch.full_like(attn_logits, float('-inf'))
                )

            attn_weights = F.softmax(attn_logits, dim=-1)  # (B, 1, N)
            attn_weights = self.dropout(attn_weights)

            # Read from values
            read = torch.matmul(attn_weights, vals)  # (B, 1, slot_dim)
            outputs.append(read)

        out = torch.cat(outputs, dim=1)  # (B, T, slot_dim)
        out = self.output_proj(out)
        return out
