from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    dropout: float = 0.1
    mlp_ratio: float = 4.0
    max_seq_len: int = 2048
    tie_embeddings: bool = True
    ffn_type: str = "swiglu"


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float, *, match_dense_params: bool) -> None:
        super().__init__()
        hidden_dim = max(1, int(mlp_ratio * hidden_size))
        if match_dense_params:
            # SwiGLU uses three projections instead of two, so we shrink the
            # hidden width when we want a near parameter-matched baseline.
            hidden_dim = max(1, int((2.0 / 3.0) * hidden_dim))
        self.up_proj = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.up_proj(x) * torch.nn.functional.silu(self.gate_proj(x))
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        hidden_dim = int(cfg.mlp_ratio * cfg.d_model)
        if cfg.ffn_type == "gelu":
            self.ffn = nn.Sequential(
                nn.Linear(cfg.d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(hidden_dim, cfg.d_model),
            )
        elif cfg.ffn_type == "swiglu":
            self.ffn = SwiGLUFFN(cfg.d_model, cfg.mlp_ratio, cfg.dropout, match_dense_params=False)
        elif cfg.ffn_type == "swiglu_matched":
            self.ffn = SwiGLUFFN(cfg.d_model, cfg.mlp_ratio, cfg.dropout, match_dense_params=True)
        else:
            raise ValueError(f"Unsupported Transformer FFN type: {cfg.ffn_type}")
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, None]:
        _, seq_len = token_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds configured max_seq_len={self.cfg.max_seq_len}.")

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        attn_mask = self._causal_mask(seq_len, token_ids.device)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, None
