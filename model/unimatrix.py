from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .deepembed import DeepEmbed
from .rosa_memory import RoSAMemoryStub


@dataclass
class UniMatrixConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    state_dim: int = 32
    dropout: float = 0.1
    mlp_ratio: float = 4.0
    deep_embed_dim: int = 128
    max_seq_len: int = 2048
    pad_token_id: int = 256
    tie_embeddings: bool = True
    use_rosa: bool = False
    use_deepembed: bool = False
    use_rule_mix: bool = False
    use_spectral_norm: bool = False
    use_step_embedding: bool = False
    use_assoc_memory: bool = False
    assoc_dim: int = 64
    assoc_topk: int = 8
    assoc_slots: int = 32
    assoc_use_token_source: bool = False
    assoc_use_identity_lookup: bool = False
    assoc_use_identity_values: bool = False
    use_sparse_pointer: bool = False
    pointer_merge_threshold: float = 0.85
    pointer_strength_floor: float = 0.05
    use_pointer_write_gate: bool = True
    use_pointer_logits: bool = False
    variant_name: str = "unimatrix-core"


ModelConfig = UniMatrixConfig


class UniMatrixBlock(nn.Module):
    def __init__(self, cfg: UniMatrixConfig):
        super().__init__()
        self.cfg = cfg
        self.inner_dim = cfg.n_heads * cfg.state_dim
        if cfg.use_sparse_pointer and not cfg.use_assoc_memory:
            raise ValueError("SparsePointer requires use_assoc_memory=True.")

        self.norm = nn.LayerNorm(cfg.d_model)
        self.ffn_norm = nn.LayerNorm(cfg.d_model)

        self.q_proj = nn.Linear(cfg.d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.inner_dim, bias=False)
        self.diag_proj = nn.Linear(cfg.d_model, self.inner_dim, bias=False)
        self.retention_proj = nn.Linear(cfg.d_model, cfg.n_heads, bias=True)
        self.output_proj = nn.Linear(self.inner_dim, cfg.d_model, bias=False)
        self.residual_gate = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.rosa_gate = nn.Linear(cfg.d_model, cfg.d_model, bias=True) if cfg.use_rosa else None
        use_identity_lookup = cfg.use_assoc_memory and cfg.assoc_use_identity_lookup and cfg.assoc_dim == cfg.d_model
        self.assoc_query_proj = (
            nn.Identity()
            if use_identity_lookup
            else (nn.Linear(cfg.d_model, cfg.assoc_dim, bias=False) if cfg.use_assoc_memory else None)
        )
        self.assoc_key_proj = (
            nn.Identity()
            if use_identity_lookup
            else (nn.Linear(cfg.d_model, cfg.assoc_dim, bias=False) if cfg.use_assoc_memory else None)
        )
        self.assoc_val_proj = (
            nn.Identity()
            if cfg.use_assoc_memory and cfg.assoc_use_identity_values
            else (nn.Linear(cfg.d_model, cfg.d_model, bias=False) if cfg.use_assoc_memory else None)
        )
        self.assoc_gate = nn.Linear(cfg.d_model, cfg.d_model, bias=True) if cfg.use_assoc_memory else None
        self.pointer_write_gate_proj = (
            nn.Linear(2 * cfg.d_model, 1, bias=True)
            if cfg.use_sparse_pointer and cfg.use_pointer_write_gate
            else None
        )
        self.pointer_logit_gate_proj = (
            nn.Linear(cfg.d_model, 1, bias=True)
            if cfg.use_sparse_pointer and cfg.use_pointer_logits
            else None
        )

        self.rule_proj = nn.Linear(cfg.d_model, 3 * cfg.n_heads, bias=True) if cfg.use_rule_mix else None
        self.dropout = nn.Dropout(cfg.dropout)

        hidden_dim = int(cfg.mlp_ratio * cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_dim, cfg.d_model),
        )

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _ = tensor.shape
        return tensor.view(batch, self.cfg.n_heads, self.cfg.state_dim)

    def _mix_rules(
        self,
        x_t: torch.Tensor,
        outer_update: torch.Tensor,
        diag_update: torch.Tensor,
        symmetric_update: torch.Tensor,
    ) -> torch.Tensor:
        if self.rule_proj is None:
            return outer_update

        logits = self.rule_proj(x_t).view(x_t.size(0), self.cfg.n_heads, 3)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1).unsqueeze(-1)
        return (
            weights[:, :, 0] * outer_update
            + weights[:, :, 1] * diag_update
            + weights[:, :, 2] * symmetric_update
        )

    def _spectral_guard(self, state: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_spectral_norm:
            return state

        frob = state.square().sum(dim=(-1, -2), keepdim=True).sqrt()
        scale = frob.clamp(min=1.0)
        return state / scale

    def _sparse_pointer_lookup(
        self,
        query: torch.Tensor,
        slot_keys: torch.Tensor,
        slot_vals: torch.Tensor,
        slot_strengths: torch.Tensor,
        slot_token_ids: Optional[torch.Tensor] = None,
        logit_gate: Optional[torch.Tensor] = None,
        vocab_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        normalized_query = F.normalize(query, dim=-1)
        normalized_keys = F.normalize(slot_keys, dim=-1)
        scores = torch.einsum("bsd,bd->bs", normalized_keys, normalized_query) / math.sqrt(slot_keys.size(-1))
        filled = slot_strengths > self.cfg.pointer_strength_floor
        scores = scores + torch.log(slot_strengths.clamp(min=1e-4))
        scores = scores.masked_fill(~filled, -1e9)
        topk = min(self.cfg.assoc_topk, scores.size(1))
        top_scores, top_indices = scores.topk(topk, dim=-1)
        gathered_values = slot_vals.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, slot_vals.size(-1)))
        weights = torch.softmax(top_scores * 8.0, dim=-1)
        assoc_context = torch.einsum("bs,bsd->bd", weights, gathered_values)

        pointer_logits = None
        if slot_token_ids is not None and vocab_size is not None:
            gathered_ids = slot_token_ids.gather(1, top_indices)
            pointer_logits = gathered_values.new_zeros(gathered_values.size(0), vocab_size)
            pointer_weights = weights
            if logit_gate is not None:
                pointer_weights = pointer_weights * logit_gate
            pointer_logits.scatter_add_(1, gathered_ids, pointer_weights)

        return assoc_context, pointer_logits

    def _sparse_pointer_write(
        self,
        candidate_key: torch.Tensor,
        candidate_val: torch.Tensor,
        write_strength: torch.Tensor,
        slot_keys: torch.Tensor,
        slot_vals: torch.Tensor,
        slot_strengths: torch.Tensor,
        slot_age: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = candidate_key.size(0)
        slot_age = slot_age + (slot_strengths > self.cfg.pointer_strength_floor).float()
        filled = slot_strengths > self.cfg.pointer_strength_floor
        normalized_candidate = F.normalize(candidate_key, dim=-1)
        normalized_keys = F.normalize(slot_keys, dim=-1)
        similarity = torch.einsum("bsd,bd->bs", normalized_keys, normalized_candidate)
        similarity = similarity.masked_fill(~filled, -1e9)
        best_similarity, best_index = similarity.max(dim=-1)
        has_empty = (~filled).any(dim=-1)
        first_empty = (~filled).float().argmax(dim=-1)
        oldest_index = slot_age.argmax(dim=-1)
        replace_index = torch.where(has_empty, first_empty, oldest_index)
        write_index = torch.where(best_similarity > self.cfg.pointer_merge_threshold, best_index, replace_index)

        batch_index = torch.arange(batch_size, device=candidate_key.device)
        slot_keys = slot_keys.clone()
        slot_vals = slot_vals.clone()
        slot_strengths = slot_strengths.clone()
        slot_age = slot_age.clone()
        slot_keys[batch_index, write_index] = candidate_key
        slot_vals[batch_index, write_index] = candidate_val
        slot_strengths[batch_index, write_index] = write_strength.squeeze(-1)
        slot_age[batch_index, write_index] = 0.0
        return slot_keys, slot_vals, slot_strengths, slot_age, write_index

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        state: torch.Tensor,
        rosa_ctx: Optional[torch.Tensor] = None,
        assoc_source: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        normalized = self.norm(x)
        outputs = []
        assoc_keys = []
        assoc_vals = []
        assoc_basis = assoc_source if assoc_source is not None else normalized
        assoc_query_all = self.assoc_query_proj(assoc_basis) if self.assoc_query_proj is not None else None
        assoc_key_all = self.assoc_key_proj(assoc_basis) if self.assoc_key_proj is not None else None
        assoc_val_all = self.assoc_val_proj(assoc_basis) if self.assoc_val_proj is not None else None
        slot_keys = None
        slot_vals = None
        slot_strengths = None
        slot_age = None
        slot_token_ids = None
        pointer_biases = [] if self.cfg.use_sparse_pointer and self.cfg.use_pointer_logits else None
        if self.cfg.use_sparse_pointer and assoc_key_all is not None and assoc_val_all is not None:
            batch_size = normalized.size(0)
            slot_keys = normalized.new_zeros(batch_size, self.cfg.assoc_slots, assoc_key_all.size(-1))
            slot_vals = normalized.new_zeros(batch_size, self.cfg.assoc_slots, assoc_val_all.size(-1))
            slot_strengths = normalized.new_zeros(batch_size, self.cfg.assoc_slots)
            slot_age = normalized.new_zeros(batch_size, self.cfg.assoc_slots)
            slot_token_ids = token_ids.new_zeros(batch_size, self.cfg.assoc_slots)

        for token_index in range(normalized.size(1)):
            x_t = normalized[:, token_index]

            q = torch.tanh(self._reshape(self.q_proj(x_t)))
            k = torch.tanh(self._reshape(self.k_proj(x_t)))
            v = torch.tanh(self._reshape(self.v_proj(x_t)))
            diag = torch.tanh(self._reshape(self.diag_proj(x_t)))

            outer_update = torch.einsum("bhd,bhe->bhde", k, v)
            diag_update = torch.diag_embed(diag)
            symmetric_update = 0.5 * (outer_update + outer_update.transpose(-1, -2))
            update = self._mix_rules(x_t, outer_update, diag_update, symmetric_update)

            retention = torch.sigmoid(self.retention_proj(x_t)).view(x_t.size(0), self.cfg.n_heads, 1, 1)
            state = retention * state + (1.0 - retention) * update
            state = self._spectral_guard(state)

            readout = torch.einsum("bhde,bhe->bhd", state, q).reshape(x_t.size(0), -1)
            mixed = self.output_proj(readout)
            mixed = torch.sigmoid(self.residual_gate(x_t)) * mixed
            pointer_logits = None

            if assoc_query_all is not None and self.assoc_gate is not None:
                if self.cfg.use_sparse_pointer and slot_keys is not None and slot_vals is not None and slot_strengths is not None:
                    if (slot_strengths > self.cfg.pointer_strength_floor).any():
                        logit_gate = None
                        if self.pointer_logit_gate_proj is not None:
                            logit_gate = torch.sigmoid(self.pointer_logit_gate_proj(x_t))
                        assoc_context, pointer_logits = self._sparse_pointer_lookup(
                            assoc_query_all[:, token_index],
                            slot_keys,
                            slot_vals,
                            slot_strengths,
                            slot_token_ids=slot_token_ids,
                            logit_gate=logit_gate,
                            vocab_size=self.cfg.vocab_size if self.cfg.use_pointer_logits else None,
                        )
                        mixed = mixed + torch.sigmoid(self.assoc_gate(x_t)) * assoc_context
                elif assoc_keys and assoc_vals:
                    query = F.normalize(assoc_query_all[:, token_index], dim=-1)
                    memory_keys = torch.stack(assoc_keys, dim=1)
                    memory_values = torch.stack(assoc_vals, dim=1)
                    normalized_keys = F.normalize(memory_keys, dim=-1)
                    scores = torch.einsum("bmd,bd->bm", normalized_keys, query) / math.sqrt(self.cfg.assoc_dim)
                    topk = min(self.cfg.assoc_topk, scores.size(1))
                    if topk < scores.size(1):
                        top_scores, top_indices = scores.topk(topk, dim=-1)
                        gathered_values = memory_values.gather(
                            1,
                            top_indices.unsqueeze(-1).expand(-1, -1, memory_values.size(-1)),
                        )
                        weights = torch.softmax(top_scores * 8.0, dim=-1)
                        assoc_context = torch.einsum("bm,bmd->bd", weights, gathered_values)
                    else:
                        weights = torch.softmax(scores * 8.0, dim=-1)
                        assoc_context = torch.einsum("bm,bmd->bd", weights, memory_values)
                    mixed = mixed + torch.sigmoid(self.assoc_gate(x_t)) * assoc_context

            if rosa_ctx is not None and self.rosa_gate is not None:
                mixed = mixed + torch.sigmoid(self.rosa_gate(x_t)) * rosa_ctx[:, token_index]

            residual = x[:, token_index] + self.dropout(mixed)
            residual = residual + self.dropout(self.ffn(self.ffn_norm(residual)))
            outputs.append(residual)
            if pointer_biases is not None:
                if pointer_logits is None:
                    pointer_logits = residual.new_zeros(residual.size(0), self.cfg.vocab_size)
                pointer_biases.append(pointer_logits)

            if assoc_key_all is not None and assoc_val_all is not None and token_index > 0:
                if self.cfg.use_sparse_pointer and slot_keys is not None and slot_vals is not None and slot_strengths is not None and slot_age is not None:
                    if self.pointer_write_gate_proj is not None:
                        write_input = torch.cat([assoc_basis[:, token_index - 1], assoc_basis[:, token_index]], dim=-1)
                        write_strength = torch.sigmoid(self.pointer_write_gate_proj(write_input))
                    else:
                        write_strength = assoc_key_all[:, token_index].new_ones((assoc_key_all.size(0), 1))
                    slot_keys, slot_vals, slot_strengths, slot_age, write_index = self._sparse_pointer_write(
                        assoc_key_all[:, token_index - 1],
                        assoc_val_all[:, token_index],
                        write_strength,
                        slot_keys,
                        slot_vals,
                        slot_strengths,
                        slot_age,
                    )
                    assert slot_token_ids is not None
                    slot_token_ids = slot_token_ids.clone()
                    batch_index = torch.arange(slot_token_ids.size(0), device=slot_token_ids.device)
                    slot_token_ids[batch_index, write_index] = token_ids[:, token_index]
                else:
                    assoc_keys.append(assoc_key_all[:, token_index - 1])
                    assoc_vals.append(assoc_val_all[:, token_index])

        out = torch.stack(outputs, dim=1)
        pointer_bias = torch.stack(pointer_biases, dim=1) if pointer_biases is not None else None
        return out, state, pointer_bias


class UniMatrixLM(nn.Module):
    def __init__(self, cfg: UniMatrixConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.step_embed = nn.Embedding(cfg.n_layers, cfg.d_model) if cfg.use_step_embedding else None
        self.block = UniMatrixBlock(cfg)
        self.deep = DeepEmbed(cfg.vocab_size, cfg.d_model, cfg.deep_embed_dim) if cfg.use_deepembed else None
        self.rosa = RoSAMemoryStub(cfg.vocab_size, cfg.d_model, pad_token_id=cfg.pad_token_id) if cfg.use_rosa else None
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        shape = (batch_size, self.cfg.n_heads, self.cfg.state_dim, self.cfg.state_dim)
        return torch.zeros(shape, device=device)

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

    def forward(
        self,
        token_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = token_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds configured max_seq_len={self.cfg.max_seq_len}.")

        if state is None:
            state = self.init_state(batch_size, token_ids.device)

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        token_features = self.token_embed(token_ids)
        x = token_features + self.pos_embed(positions)
        x = self.dropout(x)
        rosa_ctx = self.rosa(token_ids) if self.rosa is not None else None

        for step_index in range(self.cfg.n_layers):
            if self.step_embed is not None:
                x = x + self.step_embed.weight[step_index].view(1, 1, -1)
            assoc_source = token_features if self.cfg.assoc_use_token_source else None
            x, state, pointer_bias = self.block(x, token_ids, state, rosa_ctx=rosa_ctx, assoc_source=assoc_source)
            if self.deep is not None:
                x = self.deep(token_ids, x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        if pointer_bias is not None:
            logits = logits + pointer_bias
        return logits, state


def variant_config(name: str, vocab_size: int, **overrides: int | float | bool | str) -> UniMatrixConfig:
    presets: Dict[str, Dict[str, object]] = {
        "unimatrix-core": {
            "use_rosa": False,
            "use_deepembed": False,
            "use_rule_mix": False,
            "use_spectral_norm": False,
            "use_step_embedding": False,
            "use_assoc_memory": False,
        },
        "unimatrix-rosa": {
            "use_rosa": True,
            "use_deepembed": False,
            "use_rule_mix": False,
            "use_spectral_norm": False,
            "use_step_embedding": False,
            "use_assoc_memory": False,
        },
        "unimatrix-discovery": {
            "use_rosa": True,
            "use_deepembed": True,
            "use_rule_mix": True,
            "use_spectral_norm": True,
            "use_step_embedding": True,
            "use_assoc_memory": False,
        },
        "unimatrix-assoc": {
            "use_rosa": False,
            "use_deepembed": False,
            "use_rule_mix": False,
            "use_spectral_norm": False,
            "use_step_embedding": False,
            "use_assoc_memory": True,
        },
        "unimatrix-assoc-hard": {
            "use_rosa": False,
            "use_deepembed": False,
            "use_rule_mix": False,
            "use_spectral_norm": False,
            "use_step_embedding": False,
            "use_assoc_memory": True,
            "assoc_topk": 1,
        },
        "unimatrix-rosa-assoc": {
            "use_rosa": True,
            "use_deepembed": False,
            "use_rule_mix": False,
            "use_spectral_norm": False,
            "use_step_embedding": False,
            "use_assoc_memory": True,
        },
        "unimatrix-rosa-assoc-hard": {
            "use_rosa": True,
            "use_deepembed": False,
            "use_rule_mix": False,
            "use_spectral_norm": False,
            "use_step_embedding": False,
            "use_assoc_memory": True,
            "assoc_topk": 1,
        },
        "unimatrix-discovery-assoc": {
            "use_rosa": True,
            "use_deepembed": True,
            "use_rule_mix": True,
            "use_spectral_norm": True,
            "use_step_embedding": True,
            "use_assoc_memory": True,
        },
        "unimatrix-sparsepointer": {
            "use_rosa": False,
            "use_deepembed": False,
            "use_rule_mix": False,
            "use_spectral_norm": False,
            "use_step_embedding": False,
            "use_assoc_memory": True,
            "use_sparse_pointer": True,
            "assoc_use_token_source": True,
            "assoc_use_identity_lookup": True,
            "assoc_use_identity_values": True,
            "assoc_topk": 4,
            "assoc_slots": 32,
            "pointer_merge_threshold": 0.85,
            "use_pointer_write_gate": True,
            "use_pointer_logits": True,
        },
    }
    if name not in presets:
        raise ValueError(f"Unknown UniMatrix variant: {name}")

    base = {"vocab_size": vocab_size, "variant_name": name}
    base.update(presets[name])
    base.update(overrides)
    if name == "unimatrix-sparsepointer" and "assoc_dim" not in overrides:
        base["assoc_dim"] = int(base["d_model"])
    return UniMatrixConfig(**base)


class UniMatrixRosaLM(UniMatrixLM):
    def __init__(self, cfg: UniMatrixConfig):
        cfg.use_rosa = True
        if not cfg.variant_name:
            cfg.variant_name = "unimatrix-rosa"
        super().__init__(cfg)
