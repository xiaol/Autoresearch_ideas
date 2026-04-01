from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RecurrentFFNConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    recurrent_state_size: int = 128
    dropout: float = 0.1
    max_seq_len: int = 2048
    tie_embeddings: bool = True
    variant: str = "baseline"
    aux_loss_weight: float = 0.0


class SwiGLUBranch(nn.Module):
    def __init__(self, hidden_size: int, branch_size: int, dropout: float) -> None:
        super().__init__()
        branch_size = max(1, branch_size)
        self.up_proj = nn.Linear(hidden_size, branch_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, branch_size, bias=False)
        self.down_proj = nn.Linear(branch_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.up_proj(x) * F.silu(self.gate_proj(x))
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class SelectiveRecurrentFFN(nn.Module):
    def __init__(self, hidden_size: int, state_size: int, dropout: float) -> None:
        super().__init__()
        self.state_size = state_size
        self.forget_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.input_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.output_proj = nn.Linear(state_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        state = x.new_zeros(batch_size, self.state_size)
        outputs = []

        for token_index in range(seq_len):
            token_repr = x[:, token_index, :]
            forget = torch.tanh(self.forget_proj(token_repr))
            write_gate = torch.sigmoid(self.input_proj(token_repr))
            candidate = F.silu(self.value_proj(token_repr))
            state = forget * state + write_gate * candidate
            outputs.append(self.output_proj(F.silu(state)))

        return self.dropout(torch.stack(outputs, dim=1)), None


class ReadoutRecurrentFFN(nn.Module):
    def __init__(self, hidden_size: int, state_size: int, dropout: float) -> None:
        super().__init__()
        self.state_size = state_size
        self.forget_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.input_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.query_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.output_proj = nn.Linear(state_size, hidden_size, bias=False)
        self.initial_state = nn.Parameter(torch.zeros(state_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        state = self.initial_state.unsqueeze(0).expand(batch_size, -1)
        outputs = []

        for token_index in range(seq_len):
            token_repr = x[:, token_index, :]
            forget = torch.tanh(self.forget_proj(token_repr))
            write_gate = torch.sigmoid(self.input_proj(token_repr))
            candidate = F.silu(self.value_proj(token_repr))
            state = forget * state + write_gate * candidate

            readout = F.silu(self.query_proj(token_repr) * state)
            outputs.append(self.output_proj(readout))

        return self.dropout(torch.stack(outputs, dim=1)), None


class HybridRecurrentFFN(nn.Module):
    def __init__(self, hidden_size: int, budget_size: int, dropout: float) -> None:
        super().__init__()
        recurrent_size = max(1, budget_size // 2)
        local_size = max(1, budget_size - recurrent_size)

        self.forget_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.input_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.query_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.recurrent_out = nn.Linear(recurrent_size, hidden_size, bias=False)
        self.local_branch = SwiGLUBranch(hidden_size, local_size, dropout)
        self.initial_state = nn.Parameter(torch.zeros(recurrent_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        state = self.initial_state.unsqueeze(0).expand(batch_size, -1)
        outputs = []

        for token_index in range(seq_len):
            token_repr = x[:, token_index, :]
            forget = torch.tanh(self.forget_proj(token_repr))
            write_gate = torch.sigmoid(self.input_proj(token_repr))
            candidate = F.silu(self.value_proj(token_repr))
            state = forget * state + write_gate * candidate

            recurrent_readout = F.silu(self.query_proj(token_repr) * state)
            outputs.append(self.recurrent_out(recurrent_readout) + self.local_branch(token_repr))

        return self.dropout(torch.stack(outputs, dim=1)), None


class MatrixHybridRecurrentFFN(nn.Module):
    def __init__(self, hidden_size: int, budget_size: int, dropout: float) -> None:
        super().__init__()
        matrix_dim = max(8, int((max(budget_size, 8) // 2 * 2) ** 0.5))
        local_size = max(1, budget_size // 2)
        rank = 2

        self.matrix_dim = matrix_dim
        self.rank = rank
        self.decay_proj = nn.Linear(hidden_size, 1, bias=False)
        self.key_proj = nn.Linear(hidden_size, rank * matrix_dim, bias=False)
        self.value_proj = nn.Linear(hidden_size, rank * matrix_dim, bias=False)
        self.query_left_proj = nn.Linear(hidden_size, matrix_dim, bias=False)
        self.query_right_proj = nn.Linear(hidden_size, matrix_dim, bias=False)
        self.recurrent_out = nn.Linear(matrix_dim, hidden_size, bias=False)
        self.local_branch = SwiGLUBranch(hidden_size, local_size, dropout)
        self.initial_state = nn.Parameter(torch.zeros(matrix_dim, matrix_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        state = self.initial_state.unsqueeze(0).expand(batch_size, -1, -1).clone()
        outputs = []

        for token_index in range(seq_len):
            token_repr = x[:, token_index, :]
            decay = torch.sigmoid(self.decay_proj(token_repr)).view(batch_size, 1, 1)
            keys = self.key_proj(token_repr).view(batch_size, self.rank, self.matrix_dim)
            values = self.value_proj(token_repr).view(batch_size, self.rank, self.matrix_dim)
            write = torch.einsum("brm,brn->bmn", keys, values) / float(self.rank)
            state = decay * state + (1.0 - decay) * write

            query_left = self.query_left_proj(token_repr)
            query_right = self.query_right_proj(token_repr)
            read = torch.bmm(query_left.unsqueeze(1), state).squeeze(1) * query_right
            outputs.append(self.recurrent_out(F.silu(read)) + self.local_branch(token_repr))

        return self.dropout(torch.stack(outputs, dim=1)), None


class MultiTimescaleHybridRecurrentFFN(nn.Module):
    def __init__(self, hidden_size: int, budget_size: int, dropout: float) -> None:
        super().__init__()
        self.num_scales = 4
        recurrent_size = max(self.num_scales, budget_size // 2)
        self.chunk_size = max(1, recurrent_size // self.num_scales)
        self.actual_size = self.num_scales * self.chunk_size
        local_size = max(1, budget_size - self.actual_size)

        self.input_proj = nn.Linear(hidden_size, self.actual_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, self.actual_size, bias=False)
        self.query_proj = nn.Linear(hidden_size, self.actual_size, bias=False)
        self.scale_mix_proj = nn.Linear(hidden_size, self.num_scales, bias=False)
        self.recurrent_out = nn.Linear(self.actual_size, hidden_size, bias=False)
        self.local_branch = SwiGLUBranch(hidden_size, local_size, dropout)
        base_decay = torch.tensor([0.50, 0.70, 0.85, 0.95], dtype=torch.float32)
        self.logit_decay = nn.Parameter(torch.logit(base_decay).unsqueeze(-1).expand(-1, self.chunk_size).clone())
        self.initial_state = nn.Parameter(torch.zeros(self.num_scales, self.chunk_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        state = self.initial_state.unsqueeze(0).expand(batch_size, -1, -1)
        outputs = []
        base_decay = torch.sigmoid(self.logit_decay).unsqueeze(0)

        for token_index in range(seq_len):
            token_repr = x[:, token_index, :]
            write_gate = torch.sigmoid(self.input_proj(token_repr)).view(batch_size, self.num_scales, self.chunk_size)
            candidate = F.silu(self.value_proj(token_repr)).view(batch_size, self.num_scales, self.chunk_size)
            state = base_decay * state + write_gate * candidate

            query = self.query_proj(token_repr).view(batch_size, self.num_scales, self.chunk_size)
            scale_mix = torch.softmax(self.scale_mix_proj(token_repr), dim=-1).unsqueeze(-1)
            read = F.silu(query * state) * scale_mix
            outputs.append(self.recurrent_out(read.reshape(batch_size, self.actual_size)) + self.local_branch(token_repr))

        return self.dropout(torch.stack(outputs, dim=1)), None


class PredictCorrectHybridRecurrentFFN(nn.Module):
    def __init__(self, hidden_size: int, budget_size: int, dropout: float) -> None:
        super().__init__()
        recurrent_size = max(1, budget_size // 2)
        local_size = max(1, budget_size - recurrent_size)

        self.prior_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.gain_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.query_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.recurrent_out = nn.Linear(recurrent_size, hidden_size, bias=False)
        self.local_branch = SwiGLUBranch(hidden_size, local_size, dropout)
        self.initial_state = nn.Parameter(torch.zeros(recurrent_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        state = self.initial_state.unsqueeze(0).expand(batch_size, -1)
        outputs = []

        for token_index in range(seq_len):
            token_repr = x[:, token_index, :]
            prior = torch.sigmoid(self.prior_proj(token_repr))
            gain = torch.sigmoid(self.gain_proj(token_repr))
            measurement = F.silu(self.value_proj(token_repr))
            prediction = prior * state
            innovation = measurement - prediction
            state = prediction + gain * innovation

            readout = F.silu(self.query_proj(token_repr) * state)
            outputs.append(self.recurrent_out(readout) + self.local_branch(token_repr))

        return self.dropout(torch.stack(outputs, dim=1)), None


class StableDynamicsHybridRecurrentFFN(nn.Module):
    def __init__(self, hidden_size: int, budget_size: int, dropout: float) -> None:
        super().__init__()
        recurrent_size = max(2, budget_size // 2)
        if recurrent_size % 2 == 1:
            recurrent_size += 1
        local_size = max(1, budget_size - recurrent_size)
        half_size = recurrent_size // 2

        self.write_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.write_gate_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.omega_proj = nn.Linear(hidden_size, half_size, bias=False)
        self.damping_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.query_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.recurrent_out = nn.Linear(recurrent_size, hidden_size, bias=False)
        self.local_branch = SwiGLUBranch(hidden_size, local_size, dropout)
        self.initial_state = nn.Parameter(torch.zeros(recurrent_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        state = self.initial_state.unsqueeze(0).expand(batch_size, -1)
        outputs = []
        half_size = state.size(-1) // 2

        for token_index in range(seq_len):
            token_repr = x[:, token_index, :]
            even = state[:, :half_size]
            odd = state[:, half_size:]
            omega = torch.pi * torch.tanh(self.omega_proj(token_repr))
            cos_omega = torch.cos(omega)
            sin_omega = torch.sin(omega)
            rot_even = cos_omega * even - sin_omega * odd
            rot_odd = sin_omega * even + cos_omega * odd
            rotated = torch.cat([rot_even, rot_odd], dim=-1)

            damping = torch.exp(-F.softplus(self.damping_proj(token_repr)))
            write_gate = torch.sigmoid(self.write_gate_proj(token_repr))
            candidate = F.silu(self.write_proj(token_repr))
            state = damping * rotated + write_gate * candidate

            readout = F.silu(self.query_proj(token_repr) * state)
            outputs.append(self.recurrent_out(readout) + self.local_branch(token_repr))

        return self.dropout(torch.stack(outputs, dim=1)), None


class UltraCompoundRecurrentFFN(nn.Module):
    def __init__(self, hidden_size: int, budget_size: int, dropout: float) -> None:
        super().__init__()
        state_size = max(8, budget_size)
        if state_size % 2 == 1:
            state_size += 1
        half_size = state_size // 2
        local_size = max(hidden_size, 2 * budget_size)

        self.read_forget_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.read_input_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.read_value_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.read_query_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.readout_proj = nn.Linear(state_size, hidden_size, bias=False)

        self.stable_write_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.stable_gate_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.stable_omega_proj = nn.Linear(hidden_size, half_size, bias=False)
        self.stable_damping_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.stable_query_proj = nn.Linear(hidden_size, state_size, bias=False)
        self.stable_out_proj = nn.Linear(state_size, hidden_size, bias=False)

        self.local_branch = SwiGLUBranch(hidden_size, local_size, dropout)
        self.mix_proj = nn.Linear(hidden_size, 3, bias=False)
        self.read_initial_state = nn.Parameter(torch.zeros(state_size))
        self.stable_initial_state = nn.Parameter(torch.zeros(state_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        read_state = self.read_initial_state.unsqueeze(0).expand(batch_size, -1)
        stable_state = self.stable_initial_state.unsqueeze(0).expand(batch_size, -1)
        outputs = []
        half_size = stable_state.size(-1) // 2

        for token_index in range(seq_len):
            token_repr = x[:, token_index, :]

            read_forget = torch.tanh(self.read_forget_proj(token_repr))
            read_gate = torch.sigmoid(self.read_input_proj(token_repr))
            read_candidate = F.silu(self.read_value_proj(token_repr))
            read_state = read_forget * read_state + read_gate * read_candidate
            read_features = self.readout_proj(F.silu(self.read_query_proj(token_repr) * read_state))

            even = stable_state[:, :half_size]
            odd = stable_state[:, half_size:]
            omega = torch.pi * torch.tanh(self.stable_omega_proj(token_repr))
            cos_omega = torch.cos(omega)
            sin_omega = torch.sin(omega)
            rot_even = cos_omega * even - sin_omega * odd
            rot_odd = sin_omega * even + cos_omega * odd
            rotated = torch.cat([rot_even, rot_odd], dim=-1)

            stable_damping = torch.exp(-F.softplus(self.stable_damping_proj(token_repr)))
            stable_gate = torch.sigmoid(self.stable_gate_proj(token_repr))
            stable_candidate = F.silu(self.stable_write_proj(token_repr))
            stable_state = stable_damping * rotated + stable_gate * stable_candidate
            stable_features = self.stable_out_proj(F.silu(self.stable_query_proj(token_repr) * stable_state))

            local_features = self.local_branch(token_repr)
            mix_weights = torch.softmax(self.mix_proj(token_repr), dim=-1)
            mixed = (
                mix_weights[:, :1] * read_features
                + mix_weights[:, 1:2] * stable_features
                + mix_weights[:, 2:] * local_features
            )
            outputs.append(mixed)

        return self.dropout(torch.stack(outputs, dim=1)), None


class SparseWriteHybridRecurrentFFN(nn.Module):
    def __init__(self, hidden_size: int, budget_size: int, dropout: float) -> None:
        super().__init__()
        recurrent_size = max(1, budget_size // 2)
        local_size = max(1, budget_size - recurrent_size)
        self.topk = max(1, recurrent_size // 4)

        self.forget_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.write_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.query_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.recurrent_out = nn.Linear(recurrent_size, hidden_size, bias=False)
        self.local_branch = SwiGLUBranch(hidden_size, local_size, dropout)
        self.initial_state = nn.Parameter(torch.zeros(recurrent_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        state = self.initial_state.unsqueeze(0).expand(batch_size, -1)
        outputs = []

        for token_index in range(seq_len):
            token_repr = x[:, token_index, :]
            forget = torch.tanh(self.forget_proj(token_repr))
            write_scores = torch.sigmoid(self.write_proj(token_repr))
            candidate = F.silu(self.value_proj(token_repr))

            top_values, top_indices = torch.topk(write_scores, k=self.topk, dim=-1)
            del top_values
            sparse_mask = torch.zeros_like(write_scores)
            sparse_mask.scatter_(1, top_indices, 1.0)
            sparse_write = write_scores * sparse_mask
            state = forget * state + sparse_write * candidate

            readout = F.silu(self.query_proj(token_repr) * state)
            outputs.append(self.recurrent_out(readout) + self.local_branch(token_repr))

        return self.dropout(torch.stack(outputs, dim=1)), None


class AuxiliaryHybridRecurrentFFN(nn.Module):
    def __init__(self, hidden_size: int, budget_size: int, dropout: float) -> None:
        super().__init__()
        recurrent_size = max(1, budget_size // 2)
        local_size = max(1, budget_size - recurrent_size)

        self.forget_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.input_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.query_proj = nn.Linear(hidden_size, recurrent_size, bias=False)
        self.recurrent_out = nn.Linear(recurrent_size, hidden_size, bias=False)
        self.aux_head = nn.Linear(recurrent_size, hidden_size, bias=False)
        self.local_branch = SwiGLUBranch(hidden_size, local_size, dropout)
        self.initial_state = nn.Parameter(torch.zeros(recurrent_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        state = self.initial_state.unsqueeze(0).expand(batch_size, -1)
        outputs = []
        aux_losses = []

        for token_index in range(seq_len):
            token_repr = x[:, token_index, :]
            forget = torch.tanh(self.forget_proj(token_repr))
            write_gate = torch.sigmoid(self.input_proj(token_repr))
            candidate = F.silu(self.value_proj(token_repr))
            state = forget * state + write_gate * candidate

            readout = F.silu(self.query_proj(token_repr) * state)
            outputs.append(self.recurrent_out(readout) + self.local_branch(token_repr))

            if token_index + 1 < seq_len:
                aux_pred = self.aux_head(F.silu(state))
                aux_target = x[:, token_index + 1, :].detach()
                aux_losses.append(F.mse_loss(aux_pred, aux_target))

        aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        return self.dropout(torch.stack(outputs, dim=1)), aux_loss


class RecurrentFFNBlock(nn.Module):
    def __init__(self, cfg: RecurrentFFNConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        if cfg.variant == "baseline":
            self.recurrent_ffn = SelectiveRecurrentFFN(
                hidden_size=cfg.d_model,
                state_size=cfg.recurrent_state_size,
                dropout=cfg.dropout,
            )
        elif cfg.variant == "readout":
            self.recurrent_ffn = ReadoutRecurrentFFN(
                hidden_size=cfg.d_model,
                state_size=cfg.recurrent_state_size,
                dropout=cfg.dropout,
            )
        elif cfg.variant == "hybrid":
            self.recurrent_ffn = HybridRecurrentFFN(
                hidden_size=cfg.d_model,
                budget_size=cfg.recurrent_state_size,
                dropout=cfg.dropout,
            )
        elif cfg.variant == "matrix":
            self.recurrent_ffn = MatrixHybridRecurrentFFN(
                hidden_size=cfg.d_model,
                budget_size=cfg.recurrent_state_size,
                dropout=cfg.dropout,
            )
        elif cfg.variant == "multiscale":
            self.recurrent_ffn = MultiTimescaleHybridRecurrentFFN(
                hidden_size=cfg.d_model,
                budget_size=cfg.recurrent_state_size,
                dropout=cfg.dropout,
            )
        elif cfg.variant == "predictive":
            self.recurrent_ffn = PredictCorrectHybridRecurrentFFN(
                hidden_size=cfg.d_model,
                budget_size=cfg.recurrent_state_size,
                dropout=cfg.dropout,
            )
        elif cfg.variant == "stable":
            self.recurrent_ffn = StableDynamicsHybridRecurrentFFN(
                hidden_size=cfg.d_model,
                budget_size=cfg.recurrent_state_size,
                dropout=cfg.dropout,
            )
        elif cfg.variant == "ultra":
            self.recurrent_ffn = UltraCompoundRecurrentFFN(
                hidden_size=cfg.d_model,
                budget_size=cfg.recurrent_state_size,
                dropout=cfg.dropout,
            )
        elif cfg.variant == "sparse":
            self.recurrent_ffn = SparseWriteHybridRecurrentFFN(
                hidden_size=cfg.d_model,
                budget_size=cfg.recurrent_state_size,
                dropout=cfg.dropout,
            )
        elif cfg.variant == "auxiliary":
            self.recurrent_ffn = AuxiliaryHybridRecurrentFFN(
                hidden_size=cfg.d_model,
                budget_size=cfg.recurrent_state_size,
                dropout=cfg.dropout,
            )
        else:
            raise ValueError(f"Unsupported RecurrentFFN variant: {cfg.variant}")
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        ffn_out, aux_loss = self.recurrent_ffn(self.norm2(x))
        x = x + ffn_out
        return x, aux_loss


class RecurrentFFNLM(nn.Module):
    def __init__(self, cfg: RecurrentFFNConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.layers = nn.ModuleList([RecurrentFFNBlock(cfg) for _ in range(cfg.n_layers)])
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

    def get_aux_loss_weight(self) -> float:
        return self.cfg.aux_loss_weight

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = token_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds configured max_seq_len={self.cfg.max_seq_len}.")

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        attn_mask = self._causal_mask(seq_len, token_ids.device)
        aux_losses = []

        for layer in self.layers:
            x, aux_loss = layer(x, attn_mask=attn_mask)
            if aux_loss is not None:
                aux_losses.append(aux_loss)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        total_aux = torch.stack(aux_losses).mean() if aux_losses else None
        return logits, total_aux
