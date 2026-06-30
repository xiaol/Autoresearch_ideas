from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MultiStateMemoryConfig:
    d_model: int = 256
    n_heads: int = 4          # number of independent state heads
    head_dim: int = 32        # dimension per head
    max_states: int = 32      # max number of tracked states
    boundary_window: int = 8  # window for change-point detection
    state_mix: bool = True    # whether to mix states on readout
    dropout: float = 0.1


class RWKV7Head(nn.Module):
    """A single RWKV-7 style recurrent head with vector-state recurrence.

    Maintains a (head_dim,) state vector that evolves via element-wise decay:
        state_new = state * w + (1-w) * (k * v)
    and reads out as:
        y = r * state_new
    """

    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.r_proj = nn.Linear(d_model, head_dim)
        self.w_proj = nn.Linear(d_model, head_dim)  # decay
        self.k_proj = nn.Linear(d_model, head_dim)
        self.v_proj = nn.Linear(d_model, head_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, D), state: (B, head_dim)
        r = self.r_proj(x)                          # (B, head_dim)
        w = torch.sigmoid(self.w_proj(x))           # (B, head_dim), decay gate in (0,1)
        k = self.k_proj(x)                          # (B, head_dim)
        v = self.v_proj(x)                          # (B, head_dim)

        # RWKV-7 vector-state update:
        #   state_new = state * w + (1-w) * (k * v)
        # This is the diagonal-decay analogue of the matrix form
        #   S_new = diag(w) * S + (1-w) * (k v^T)
        new_state = state * w + (1 - w) * (k * v)   # (B, head_dim)

        # Readout: y = r * state_new (element-wise)
        y = r * new_state                           # (B, head_dim)

        return y, new_state


class ChangePointDetector(nn.Module):
    """Adaptive boundary detection: detects shifts in token variance."""

    def __init__(self, d_model: int, window: int = 8):
        super().__init__()
        self.window = window
        self.detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        window = min(self.window, T)

        # Compute running variance
        running_mean = torch.cumsum(x, dim=1) / torch.arange(1, T + 1, device=x.device).float().view(1, -1, 1)
        variance = ((x - running_mean) ** 2).mean(dim=-1)  # (B, T)

        # Detect shifts: high variance change = boundary
        if T > 1:
            variance_change = torch.abs(variance[:, 1:] - variance[:, :-1])  # (B, T-1)
            variance_change = F.pad(variance_change, (0, 1), value=0)  # (B, T)
        else:
            variance_change = torch.zeros_like(variance)

        # Normalize
        vc_max = variance_change.max(dim=-1, keepdim=True).values + 1e-8
        boundary_scores = variance_change / vc_max  # (B, T)

        return boundary_scores.unsqueeze(-1)  # (B, T, 1)


class MultiStateMemory(nn.Module):
    """Multi-head online memory with adaptive state boundaries.

    Maintains n_heads independent recurrent states. Uses change-point detection
    to segment the sequence and assign different state segments per head.
    """

    def __init__(self, config: MultiStateMemoryConfig):
        super().__init__()
        self.config = config

        # Create independent RWKV-7 heads
        self.heads = nn.ModuleList([
            RWKV7Head(config.d_model, config.head_dim)
            for _ in range(config.n_heads)
        ])

        # Boundary detector
        self.boundary_detector = ChangePointDetector(config.d_model, config.boundary_window)

        # State projection
        self.state_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model)

        # Readout mixing (if enabled)
        if config.state_mix:
            self.mix_gate = nn.Sequential(
                nn.Linear(config.d_model + config.n_heads * config.head_dim, config.n_heads),
                nn.Sigmoid(),
            )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, boundary_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        device = x.device

        # Detect boundaries (or use provided mask)
        if boundary_mask is None:
            boundary_scores = self.boundary_detector(x)  # (B, T, 1)
            boundaries = (boundary_scores > 0.6).float()
        else:
            boundaries = boundary_mask  # (B, T, 1)

        # Initialize states per head per batch
        states = [torch.zeros(B, self.config.head_dim, device=device) for _ in range(self.config.n_heads)]

        outputs = []

        for t in range(T):
            xt = x[:, t, :]  # (B, D)

            # Check for boundary reset
            if t > 0:
                reset = boundaries[:, t, :]  # (B, 1)
                for i in range(self.config.n_heads):
                    # Reset state where boundary detected
                    states[i] = states[i] * (1 - reset)

            # Run each head
            head_outputs = []
            for i, head in enumerate(self.heads):
                h_out, new_state = head(xt, states[i])
                head_outputs.append(h_out)          # (B, head_dim)
                states[i] = new_state

            # Concatenate heads
            combined = torch.cat(head_outputs, dim=-1)  # (B, n_heads * head_dim)

            # Mixing gate (if enabled): attend over heads per token
            if self.config.state_mix:
                mix_input = torch.cat([xt, combined], dim=-1)  # (B, D + n_heads*head_dim)
                mix_weights = self.mix_gate(mix_input)         # (B, n_heads)
                combined_reshaped = combined.view(B, self.config.n_heads, self.config.head_dim)
                mixed = (combined_reshaped * mix_weights.unsqueeze(-1)).reshape(B, -1)
            else:
                mixed = combined

            # Project to d_model
            out = self.state_proj(mixed)  # (B, D)
            outputs.append(out.unsqueeze(1))

        output = torch.cat(outputs, dim=1)  # (B, T, D)
        output = self.dropout(output)

        return output
