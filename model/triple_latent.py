from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TripleLatentConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    state_dim: int = 32
    num_slots: int = 8
    dropout: float = 0.1
    mlp_ratio: float = 4.0
    max_seq_len: int = 2048
    tie_embeddings: bool = True
    use_slots: bool = False
    use_local_conv: bool = False
    use_assoc_memory: bool = False
    assoc_dim: int = 64
    assoc_topk: int = 4
    assoc_fusion: str = "sum"
    assoc_query_mode: str = "parallel"
    assoc_layer_mode: str = "all"
    use_assoc_read_gate: bool = False
    use_assoc_write_gate: bool = False
    variant_name: str = "triple-latent"


class TripleLatentLayer(nn.Module):
    def __init__(self, cfg: TripleLatentConfig):
        super().__init__()
        self.cfg = cfg
        inner_dim = cfg.n_heads * cfg.state_dim

        self.norm = nn.LayerNorm(cfg.d_model)
        self.ffn_norm = nn.LayerNorm(cfg.d_model)
        self.a_proj = nn.Linear(cfg.d_model, inner_dim, bias=False)
        self.b_proj = nn.Linear(cfg.d_model, inner_dim, bias=False)
        self.q_left_proj = nn.Linear(cfg.d_model, inner_dim, bias=False)
        self.q_right_proj = nn.Linear(cfg.d_model, inner_dim, bias=False)
        self.state_decay_proj = nn.Linear(cfg.d_model, cfg.n_heads, bias=True)
        self.pair_decay_proj = nn.Linear(cfg.d_model, cfg.n_heads, bias=True)

        if cfg.use_slots:
            self.slot_gate_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.num_slots, bias=True)
            output_width = cfg.n_heads * cfg.num_slots
        else:
            self.slot_gate_proj = None
            output_width = inner_dim

        self.output_proj = nn.Linear(output_width, cfg.d_model, bias=False)
        self.assoc_query_proj = nn.Linear(cfg.d_model, cfg.assoc_dim, bias=False) if cfg.use_assoc_memory else None
        self.assoc_key_proj = nn.Linear(cfg.d_model, cfg.assoc_dim, bias=False) if cfg.use_assoc_memory else None
        self.assoc_val_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False) if cfg.use_assoc_memory else None
        self.assoc_read_gate_proj = (
            nn.Linear(cfg.d_model, 1, bias=True) if cfg.use_assoc_memory and cfg.use_assoc_read_gate else None
        )
        self.assoc_write_gate_proj = (
            nn.Linear(2 * cfg.d_model, 1, bias=True) if cfg.use_assoc_memory and cfg.use_assoc_write_gate else None
        )
        self.concat_output_proj = (
            nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
            if cfg.use_assoc_memory and cfg.assoc_fusion == "concat"
            else None
        )
        hidden_dim = int(cfg.mlp_ratio * cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_dim, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def _assoc_lookup(
        self,
        *,
        query_basis: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        memory_strengths: torch.Tensor,
    ) -> torch.Tensor:
        assert self.assoc_query_proj is not None

        query = F.normalize(self.assoc_query_proj(query_basis), dim=-1)
        normalized_keys = F.normalize(memory_keys, dim=-1)
        scores = torch.einsum("bmd,bd->bm", normalized_keys, query) / math.sqrt(self.cfg.assoc_dim)
        scores = scores + torch.log(memory_strengths.squeeze(-1).clamp(min=1e-4))
        topk = min(self.cfg.assoc_topk, scores.size(1))
        if topk < scores.size(1):
            top_scores, top_indices = scores.topk(topk, dim=-1)
            gathered_values = memory_values.gather(
                1,
                top_indices.unsqueeze(-1).expand(-1, -1, memory_values.size(-1)),
            )
            weights = torch.softmax(top_scores * 8.0, dim=-1)
            return torch.einsum("bm,bmd->bd", weights, gathered_values)

        weights = torch.softmax(scores * 8.0, dim=-1)
        return torch.einsum("bm,bmd->bd", weights, memory_values)

    def forward(
        self,
        x: torch.Tensor,
        assoc_source: Optional[torch.Tensor] = None,
        *,
        enable_assoc: bool = False,
        return_assoc: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        normalized = self.norm(x)
        batch_size, seq_len, _ = normalized.shape
        state = normalized.new_zeros(batch_size, self.cfg.n_heads, self.cfg.state_dim)
        pair_memory = None if self.cfg.use_slots else normalized.new_zeros(
            batch_size,
            self.cfg.n_heads,
            self.cfg.state_dim,
            self.cfg.state_dim,
        )
        slot_left = None
        slot_right = None
        if self.cfg.use_slots:
            slot_left = normalized.new_zeros(batch_size, self.cfg.n_heads, self.cfg.num_slots, self.cfg.state_dim)
            slot_right = normalized.new_zeros(batch_size, self.cfg.n_heads, self.cfg.num_slots, self.cfg.state_dim)

        outputs = normalized.new_empty(batch_size, seq_len, self.cfg.d_model)
        assoc_outputs = normalized.new_zeros(batch_size, seq_len, self.cfg.d_model) if return_assoc else None
        a_all = torch.tanh(self.a_proj(normalized)).view(batch_size, seq_len, self.cfg.n_heads, self.cfg.state_dim)
        b_all = torch.tanh(self.b_proj(normalized)).view(batch_size, seq_len, self.cfg.n_heads, self.cfg.state_dim)
        q_left_all = torch.tanh(self.q_left_proj(normalized)).view(
            batch_size,
            seq_len,
            self.cfg.n_heads,
            self.cfg.state_dim,
        )
        q_right_all = torch.tanh(self.q_right_proj(normalized)).view(
            batch_size,
            seq_len,
            self.cfg.n_heads,
            self.cfg.state_dim,
        )
        state_decay_all = torch.sigmoid(self.state_decay_proj(normalized)).view(
            batch_size,
            seq_len,
            self.cfg.n_heads,
            1,
        )
        pair_decay_all = torch.sigmoid(self.pair_decay_proj(normalized)).view(
            batch_size,
            seq_len,
            self.cfg.n_heads,
            1,
            1,
        )
        slot_gate_all = None
        if self.cfg.use_slots and self.slot_gate_proj is not None:
            slot_gate_all = torch.softmax(
                self.slot_gate_proj(normalized).view(batch_size, seq_len, self.cfg.n_heads, self.cfg.num_slots),
                dim=-1,
            ).unsqueeze(-1)

        assoc_basis = assoc_source if assoc_source is not None else normalized
        assoc_key_all = None
        assoc_val_all = None
        assoc_memory_keys = None
        assoc_memory_vals = None
        assoc_memory_strengths = None
        if enable_assoc and self.assoc_key_proj is not None and self.assoc_val_proj is not None:
            assoc_key_all = self.assoc_key_proj(assoc_basis)
            assoc_val_all = self.assoc_val_proj(assoc_basis)
            if seq_len > 1:
                assoc_memory_keys = assoc_key_all[:, :-1]
                assoc_memory_vals = assoc_val_all[:, 1:]
                assoc_memory_strengths = normalized.new_ones(batch_size, seq_len - 1, 1)
                if self.assoc_write_gate_proj is not None:
                    write_basis = torch.cat([assoc_basis[:, :-1], assoc_basis[:, 1:]], dim=-1)
                    assoc_memory_strengths = torch.sigmoid(self.assoc_write_gate_proj(write_basis))
                    assoc_memory_vals = assoc_memory_vals * assoc_memory_strengths

        for token_index in range(seq_len):
            a = a_all[:, token_index]
            b = b_all[:, token_index]
            q_left = q_left_all[:, token_index]
            q_right = q_right_all[:, token_index]
            state_decay = state_decay_all[:, token_index]
            pair_decay = pair_decay_all[:, token_index]

            previous_state = state
            state = state_decay * state + (1.0 - state_decay) * a

            if self.cfg.use_slots:
                assert slot_left is not None and slot_right is not None and slot_gate_all is not None
                slot_gate = slot_gate_all[:, token_index]
                slot_left = pair_decay * slot_left + (1.0 - pair_decay) * slot_gate * previous_state.unsqueeze(2)
                slot_right = pair_decay * slot_right + (1.0 - pair_decay) * slot_gate * b.unsqueeze(2)
                left_scores = (q_left.unsqueeze(2) * slot_left).sum(dim=-1)
                right_scores = (q_right.unsqueeze(2) * slot_right).sum(dim=-1)
                mixed = self.output_proj((left_scores * right_scores).reshape(batch_size, -1))
            else:
                assert pair_memory is not None
                pair_memory = pair_decay * pair_memory + (1.0 - pair_decay) * torch.einsum(
                    "bhd,bhe->bhde",
                    previous_state,
                    b,
                )
                left_context = torch.einsum("bhde,bhd->bhe", pair_memory, q_left)
                mixed = self.output_proj((left_context * q_right).reshape(batch_size, -1))

            base_residual = x[:, token_index] + self.dropout(mixed)
            assoc_context = None
            available_memory = token_index - 1
            if (
                enable_assoc
                and assoc_memory_keys is not None
                and assoc_memory_vals is not None
                and assoc_memory_strengths is not None
                and available_memory > 0
            ):
                query_basis = normalized[:, token_index]
                if self.cfg.assoc_query_mode == "serial":
                    query_basis = base_residual
                elif self.cfg.assoc_query_mode == "previous" and token_index > 0:
                    query_basis = assoc_basis[:, token_index - 1]
                assoc_context = self._assoc_lookup(
                    query_basis=query_basis,
                    memory_keys=assoc_memory_keys[:, :available_memory],
                    memory_values=assoc_memory_vals[:, :available_memory],
                    memory_strengths=assoc_memory_strengths[:, :available_memory],
                )
                if self.assoc_read_gate_proj is not None:
                    assoc_context = torch.sigmoid(self.assoc_read_gate_proj(query_basis)) * assoc_context

            if assoc_outputs is not None:
                assoc_outputs[:, token_index] = (
                    base_residual.new_zeros(batch_size, self.cfg.d_model)
                    if assoc_context is None
                    else assoc_context
                )

            residual = base_residual
            if enable_assoc and self.cfg.assoc_fusion != "logits":
                assoc_term = assoc_context if assoc_context is not None else base_residual.new_zeros(batch_size, self.cfg.d_model)
                if self.cfg.assoc_fusion == "sum":
                    residual = base_residual + self.dropout(assoc_term)
                elif self.cfg.assoc_fusion == "concat":
                    assert self.concat_output_proj is not None
                    fused = self.concat_output_proj(torch.cat([mixed, assoc_term], dim=-1))
                    residual = x[:, token_index] + self.dropout(fused)

            residual = residual + self.dropout(self.ffn(self.ffn_norm(residual)))
            outputs[:, token_index] = residual

        return outputs, assoc_outputs


class TripleLatentLM(nn.Module):
    def __init__(self, cfg: TripleLatentConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.local_conv = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=5, padding=2) if cfg.use_local_conv else None
        self.layers = nn.ModuleList([TripleLatentLayer(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.assoc_norm = nn.LayerNorm(cfg.d_model) if cfg.use_assoc_memory and cfg.assoc_fusion == "logits" else None
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, token_ids: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        del state
        _, seq_len = token_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds configured max_seq_len={self.cfg.max_seq_len}.")

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        token_features = self.token_embed(token_ids)
        x = token_features + self.pos_embed(positions)
        x = self.dropout(x)

        if self.local_conv is not None:
            conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
            x = x + self.dropout(F.gelu(conv))

        assoc_context = None
        assoc_count = 0
        for layer_index, layer in enumerate(self.layers):
            enable_assoc = self.cfg.use_assoc_memory and (
                self.cfg.assoc_layer_mode == "all" or layer_index == len(self.layers) - 1
            )
            collect_assoc = enable_assoc and self.cfg.assoc_fusion == "logits"
            x, layer_assoc = layer(
                x,
                assoc_source=token_features,
                enable_assoc=enable_assoc,
                return_assoc=collect_assoc,
            )
            if layer_assoc is not None:
                assoc_context = layer_assoc if assoc_context is None else assoc_context + layer_assoc
                assoc_count += 1

        x = self.final_norm(x)
        logits = self.lm_head(x)
        if assoc_context is not None and assoc_count > 0 and self.assoc_norm is not None:
            logits = logits + self.lm_head(self.assoc_norm(assoc_context / assoc_count))
        return logits, None


def triple_latent_config(name: str, vocab_size: int, **overrides: int | float | bool | str) -> TripleLatentConfig:
    presets = {
        "triple-latent": {
            "use_slots": False,
            "use_local_conv": False,
        },
        "triple-slot": {
            "use_slots": True,
            "use_local_conv": False,
        },
        "triple-hybrid": {
            "use_slots": False,
            "use_local_conv": True,
        },
        "triple-slot-hybrid": {
            "use_slots": True,
            "use_local_conv": True,
        },
        "triple-hybrid-assoc-sum": {
            "use_slots": False,
            "use_local_conv": True,
            "use_assoc_memory": True,
            "assoc_fusion": "sum",
            "assoc_query_mode": "parallel",
            "assoc_layer_mode": "all",
        },
        "triple-hybrid-assoc-concat": {
            "use_slots": False,
            "use_local_conv": True,
            "use_assoc_memory": True,
            "assoc_fusion": "concat",
            "assoc_query_mode": "parallel",
            "assoc_layer_mode": "all",
        },
        "triple-hybrid-assoc-logits": {
            "use_slots": False,
            "use_local_conv": True,
            "use_assoc_memory": True,
            "assoc_fusion": "logits",
            "assoc_query_mode": "parallel",
            "assoc_layer_mode": "all",
        },
        "triple-hybrid-assoc-serial-sum": {
            "use_slots": False,
            "use_local_conv": True,
            "use_assoc_memory": True,
            "assoc_fusion": "sum",
            "assoc_query_mode": "serial",
            "assoc_layer_mode": "all",
        },
        "triple-hybrid-assoc-last-sum": {
            "use_slots": False,
            "use_local_conv": True,
            "use_assoc_memory": True,
            "assoc_fusion": "sum",
            "assoc_query_mode": "parallel",
            "assoc_layer_mode": "last",
        },
        "triple-hybrid-assoc-last-logits": {
            "use_slots": False,
            "use_local_conv": True,
            "use_assoc_memory": True,
            "assoc_fusion": "logits",
            "assoc_query_mode": "parallel",
            "assoc_layer_mode": "last",
        },
        "triple-hybrid-assoc-prev-sum": {
            "use_slots": False,
            "use_local_conv": True,
            "use_assoc_memory": True,
            "assoc_fusion": "sum",
            "assoc_query_mode": "previous",
            "assoc_layer_mode": "all",
        },
        "triple-hybrid-assoc-prev-logits": {
            "use_slots": False,
            "use_local_conv": True,
            "use_assoc_memory": True,
            "assoc_fusion": "logits",
            "assoc_query_mode": "previous",
            "assoc_layer_mode": "all",
        },
        "triple-hybrid-assoc-gated-last-logits": {
            "use_slots": False,
            "use_local_conv": True,
            "use_assoc_memory": True,
            "assoc_fusion": "logits",
            "assoc_query_mode": "previous",
            "assoc_layer_mode": "last",
            "assoc_topk": 8,
            "use_assoc_read_gate": True,
        },
        "triple-hybrid-assoc-writegated-last-logits": {
            "use_slots": False,
            "use_local_conv": True,
            "use_assoc_memory": True,
            "assoc_fusion": "logits",
            "assoc_query_mode": "previous",
            "assoc_layer_mode": "last",
            "assoc_topk": 8,
            "use_assoc_read_gate": True,
            "use_assoc_write_gate": True,
        },
        "triple-hybrid-assoc-writegated-last-sum": {
            "use_slots": False,
            "use_local_conv": True,
            "use_assoc_memory": True,
            "assoc_fusion": "sum",
            "assoc_query_mode": "previous",
            "assoc_layer_mode": "last",
            "assoc_topk": 8,
            "use_assoc_read_gate": True,
            "use_assoc_write_gate": True,
        },
        "triple-hybrid-assoc-writegated-all-logits": {
            "use_slots": False,
            "use_local_conv": True,
            "use_assoc_memory": True,
            "assoc_fusion": "logits",
            "assoc_query_mode": "previous",
            "assoc_layer_mode": "all",
            "assoc_topk": 8,
            "use_assoc_read_gate": True,
            "use_assoc_write_gate": True,
        },
    }
    if name not in presets:
        raise ValueError(f"Unknown triple-latent variant: {name}")

    base = {"vocab_size": vocab_size, "variant_name": name}
    base.update(presets[name])
    base.update(overrides)
    return TripleLatentConfig(**base)
