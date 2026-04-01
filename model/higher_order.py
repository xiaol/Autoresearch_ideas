from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


HIGHER_ORDER_MODELS = [
    "transformer-triple",
    "exact-triplet",
    "pair-state",
    "pair-slot",
    "hybrid-pair",
    "typed-latent",
]


@dataclass
class HigherOrderConfig:
    vocab_size: int
    num_classes: int
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    state_dim: int = 16
    num_slots: int = 8
    dropout: float = 0.1
    max_seq_len: int = 256
    variant_name: str = "pair-state"
    query_token_id: int = 2
    role_a_token_id: int = 3
    role_b_token_id: int = 4
    role_c_token_id: int = 5
    a_start: int = 22
    b_start: int = 30
    c_start: int = 38
    tag_start: int = 46
    task_start: int = 49
    num_tags: int = 3
    num_values: int = 8
    num_tasks: int = 4


class BaseHigherOrderModel(nn.Module):
    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TokenEncoder(nn.Module):
    def __init__(self, cfg: HigherOrderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.cfg.max_seq_len}.")
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        return self.dropout(self.token_embed(token_ids) + self.pos_embed(positions))


class TransformerTripleClassifier(BaseHigherOrderModel):
    def __init__(self, cfg: HigherOrderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TokenEncoder(cfg)
        hidden_dim = int(4.0 * cfg.d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=hidden_dim,
                    dropout=cfg.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.encoder(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x[:, -1])

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(token_ids))


class ExactTripletClassifier(BaseHigherOrderModel):
    def __init__(self, cfg: HigherOrderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TokenEncoder(cfg)
        hidden_dim = int(2.0 * cfg.d_model)
        self.stem = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(hidden_dim, cfg.d_model),
                )
                for _ in range(max(cfg.n_layers - 1, 1))
            ]
        )
        self.role_norm = nn.LayerNorm(cfg.d_model)
        self.a_proj = nn.Linear(cfg.d_model, cfg.state_dim, bias=False)
        self.b_proj = nn.Linear(cfg.d_model, cfg.state_dim, bias=False)
        self.c_proj = nn.Linear(cfg.d_model, cfg.state_dim, bias=False)
        self.class_a = nn.Parameter(torch.empty(cfg.num_classes, cfg.state_dim))
        self.class_b = nn.Parameter(torch.empty(cfg.num_classes, cfg.state_dim))
        self.class_c = nn.Parameter(torch.empty(cfg.num_classes, cfg.state_dim))
        self.query_norm = nn.LayerNorm(cfg.d_model)
        self.query_head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.apply(self._init_weights)
        nn.init.normal_(self.class_a, mean=0.0, std=0.02)
        nn.init.normal_(self.class_b, mean=0.0, std=0.02)
        nn.init.normal_(self.class_c, mean=0.0, std=0.02)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _ordered_triplet_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        prefix_len = max(seq_len - 1, 1)
        idx = torch.arange(prefix_len, device=device)
        mask = (idx[:, None, None] < idx[None, :, None]) & (idx[None, :, None] < idx[None, None, :])
        return mask.float()

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.encoder(token_ids)
        for block in self.stem:
            x = x + block(x)
        return self.query_norm(x[:, -1])

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.encoder(token_ids)
        for block in self.stem:
            x = x + block(x)
        prefix = self.role_norm(x[:, :-1])
        a = torch.tanh(self.a_proj(prefix))
        b = torch.tanh(self.b_proj(prefix))
        c = torch.tanh(self.c_proj(prefix))

        ua = torch.einsum("blr,cr->blc", a, self.class_a)
        ub = torch.einsum("blr,cr->blc", b, self.class_b)
        uc = torch.einsum("blr,cr->blc", c, self.class_c)

        scores = ua[:, :, None, None, :] * ub[:, None, :, None, :] * uc[:, None, None, :, :]
        mask = self._ordered_triplet_mask(token_ids.size(1), token_ids.device)
        denom = mask.sum().clamp(min=1.0)
        logits = (scores * mask.view(1, mask.size(0), mask.size(1), mask.size(2), 1)).sum(dim=(1, 2, 3)) / denom
        return logits + self.query_head(self.query_norm(x[:, -1]))


class PairStateLayer(nn.Module):
    def __init__(self, cfg: HigherOrderConfig, *, use_slots: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_slots = use_slots
        inner_dim = cfg.n_heads * cfg.state_dim

        self.norm = nn.LayerNorm(cfg.d_model)
        self.ffn_norm = nn.LayerNorm(cfg.d_model)
        self.a_proj = nn.Linear(cfg.d_model, inner_dim, bias=False)
        self.b_proj = nn.Linear(cfg.d_model, inner_dim, bias=False)
        self.q_left_proj = nn.Linear(cfg.d_model, inner_dim, bias=False)
        self.q_right_proj = nn.Linear(cfg.d_model, inner_dim, bias=False)
        self.state_decay_proj = nn.Linear(cfg.d_model, cfg.n_heads, bias=True)
        self.pair_decay_proj = nn.Linear(cfg.d_model, cfg.n_heads, bias=True)
        if use_slots:
            self.slot_gate_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.num_slots, bias=True)
            output_width = cfg.n_heads * cfg.num_slots
        else:
            self.slot_gate_proj = None
            output_width = inner_dim
        self.output_proj = nn.Linear(output_width, cfg.d_model, bias=False)
        hidden_dim = int(4.0 * cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_dim, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.size(0), self.cfg.n_heads, self.cfg.state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        batch_size, seq_len, _ = normalized.shape
        state = normalized.new_zeros(batch_size, self.cfg.n_heads, self.cfg.state_dim)
        pair_memory = None if self.use_slots else normalized.new_zeros(batch_size, self.cfg.n_heads, self.cfg.state_dim, self.cfg.state_dim)
        slot_left = normalized.new_zeros(batch_size, self.cfg.n_heads, self.cfg.num_slots, self.cfg.state_dim) if self.use_slots else None
        slot_right = normalized.new_zeros(batch_size, self.cfg.n_heads, self.cfg.num_slots, self.cfg.state_dim) if self.use_slots else None
        outputs = []

        a_all = torch.tanh(self.a_proj(normalized)).view(batch_size, seq_len, self.cfg.n_heads, self.cfg.state_dim)
        b_all = torch.tanh(self.b_proj(normalized)).view(batch_size, seq_len, self.cfg.n_heads, self.cfg.state_dim)
        q_left_all = torch.tanh(self.q_left_proj(normalized)).view(batch_size, seq_len, self.cfg.n_heads, self.cfg.state_dim)
        q_right_all = torch.tanh(self.q_right_proj(normalized)).view(batch_size, seq_len, self.cfg.n_heads, self.cfg.state_dim)
        state_decay_all = torch.sigmoid(self.state_decay_proj(normalized)).view(batch_size, seq_len, self.cfg.n_heads, 1)
        pair_decay_all = torch.sigmoid(self.pair_decay_proj(normalized)).view(batch_size, seq_len, self.cfg.n_heads, 1, 1)
        slot_gate_all = None
        if self.use_slots and self.slot_gate_proj is not None:
            slot_gate_all = torch.softmax(
                self.slot_gate_proj(normalized).view(batch_size, seq_len, self.cfg.n_heads, self.cfg.num_slots),
                dim=-1,
            ).unsqueeze(-1)

        for token_index in range(seq_len):
            a = a_all[:, token_index]
            b = b_all[:, token_index]
            q_left = q_left_all[:, token_index]
            q_right = q_right_all[:, token_index]
            state_decay = state_decay_all[:, token_index]
            pair_decay = pair_decay_all[:, token_index]

            previous_state = state
            state = state_decay * state + (1.0 - state_decay) * a

            if self.use_slots and slot_left is not None and slot_right is not None and slot_gate_all is not None:
                slot_gate = slot_gate_all[:, token_index]
                slot_left = pair_decay * slot_left + (1.0 - pair_decay) * slot_gate * previous_state.unsqueeze(2)
                slot_right = pair_decay * slot_right + (1.0 - pair_decay) * slot_gate * b.unsqueeze(2)
                left_scores = (q_left.unsqueeze(2) * slot_left).sum(dim=-1)
                right_scores = (q_right.unsqueeze(2) * slot_right).sum(dim=-1)
                mixed = self.output_proj((left_scores * right_scores).reshape(batch_size, -1))
            else:
                assert pair_memory is not None
                pair_memory = pair_decay * pair_memory + (1.0 - pair_decay) * torch.einsum("bhd,bhe->bhde", previous_state, b)
                left_context = torch.einsum("bhde,bhd->bhe", pair_memory, q_left)
                mixed = self.output_proj((left_context * q_right).reshape(batch_size, -1))

            residual = x[:, token_index] + self.dropout(mixed)
            residual = residual + self.dropout(self.ffn(self.ffn_norm(residual)))
            outputs.append(residual)

        return torch.stack(outputs, dim=1)


class PairStateClassifier(BaseHigherOrderModel):
    def __init__(self, cfg: HigherOrderConfig, *, use_slots: bool = False, use_local_conv: bool = False):
        super().__init__()
        self.cfg = cfg
        self.encoder = TokenEncoder(cfg)
        self.use_local_conv = use_local_conv
        self.local_conv = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=5, padding=2) if use_local_conv else None
        self.layers = nn.ModuleList([PairStateLayer(cfg, use_slots=use_slots) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.dropout = nn.Dropout(cfg.dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.encoder(token_ids)
        if self.local_conv is not None:
            conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
            x = x + self.dropout(F.gelu(conv))
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x[:, -1])

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(token_ids))


class TypedLatentClassifier(BaseHigherOrderModel):
    def __init__(self, cfg: HigherOrderConfig):
        super().__init__()
        self.cfg = cfg
        feature_dim = cfg.num_tasks + (3 * cfg.num_values)
        self.input_proj = nn.Linear(feature_dim, cfg.d_model)
        self.feature_norm = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, max(cfg.d_model, 64)),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(max(cfg.d_model, 64), cfg.d_model),
        )
        self.lookup_table = nn.Parameter(torch.empty(cfg.num_values, cfg.num_values, cfg.num_values, cfg.num_classes))
        self.apply(self._init_weights)
        nn.init.zeros_(self.lookup_table)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _scatter_role_values(
        self,
        token_ids: torch.Tensor,
        *,
        role_token_id: int,
        value_start: int,
    ) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        memory = token_ids.new_zeros((batch_size, self.cfg.num_tags, self.cfg.num_values), dtype=torch.float32)
        if seq_len < 2:
            return memory

        latest_tag = token_ids.new_full((batch_size, seq_len), -1)
        current_tag = token_ids.new_full((batch_size,), -1)
        for position in range(seq_len):
            token = token_ids[:, position]
            is_tag = (token >= self.cfg.tag_start) & (token < self.cfg.tag_start + self.cfg.num_tags)
            current_tag = torch.where(is_tag, token - self.cfg.tag_start, current_tag)
            latest_tag[:, position] = current_tag

        current = token_ids[:, :-1]
        nxt = token_ids[:, 1:]
        tag_index_tensor = latest_tag[:, :-1]
        is_role = current == role_token_id
        is_value = (nxt >= value_start) & (nxt < value_start + self.cfg.num_values)
        has_tag = tag_index_tensor >= 0
        match = is_role & is_value & has_tag
        if not match.any():
            return memory

        batch_index, position_index = torch.nonzero(match, as_tuple=True)
        del position_index
        tag_index = tag_index_tensor[match].long()
        value_index = (nxt[match] - value_start).long()
        memory[batch_index, tag_index, value_index] = 1.0
        return memory

    def _extract_task_and_values(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = token_ids.size(0)
        device = token_ids.device
        query_tag = (token_ids[:, -1] - self.cfg.tag_start).clamp(min=0, max=self.cfg.num_tags - 1).long()
        task_index = (token_ids[:, 1] - self.cfg.task_start).clamp(min=0, max=self.cfg.num_tasks - 1).long()

        a_memory = self._scatter_role_values(
            token_ids,
            role_token_id=self.cfg.role_a_token_id,
            value_start=self.cfg.a_start,
        )
        b_memory = self._scatter_role_values(
            token_ids,
            role_token_id=self.cfg.role_b_token_id,
            value_start=self.cfg.b_start,
        )
        c_memory = self._scatter_role_values(
            token_ids,
            role_token_id=self.cfg.role_c_token_id,
            value_start=self.cfg.c_start,
        )

        batch_index = torch.arange(batch_size, device=device)
        a = a_memory[batch_index, query_tag]
        b = b_memory[batch_index, query_tag]
        c = c_memory[batch_index, query_tag]
        task = F.one_hot(task_index, num_classes=self.cfg.num_tasks).float()
        return task, a, b, c

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        task, a, b, c = self._extract_task_and_values(token_ids)

        features = torch.cat([task, a, b, c], dim=-1)
        encoded = self.feature_norm(self.input_proj(features))
        return encoded + self.mlp(encoded)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        task, a, b, c = self._extract_task_and_values(token_ids)
        a_idx = a.argmax(dim=-1)
        b_idx = b.argmax(dim=-1)
        c_idx = c.argmax(dim=-1)

        copy_idx = a_idx
        affine_idx = (a_idx + 2 * b_idx + 3 * c_idx) % self.cfg.num_classes
        gate_idx = ((a_idx * (b_idx + 1)) + c_idx * ((a_idx ^ b_idx) + 1)) % self.cfg.num_classes
        lookup_logits = self.lookup_table[a_idx, b_idx, c_idx]

        logit_scale = 12.0
        copy_logits = F.one_hot(copy_idx, num_classes=self.cfg.num_classes).float() * logit_scale
        affine_logits = F.one_hot(affine_idx, num_classes=self.cfg.num_classes).float() * logit_scale
        gate_logits = F.one_hot(gate_idx, num_classes=self.cfg.num_classes).float() * logit_scale
        stacked = torch.stack([copy_logits, affine_logits, gate_logits, lookup_logits], dim=1)
        return torch.einsum("bt,btc->bc", task, stacked)


def build_higher_order_model(name: str, cfg: HigherOrderConfig) -> BaseHigherOrderModel:
    if name == "transformer-triple":
        return TransformerTripleClassifier(cfg)
    if name == "exact-triplet":
        return ExactTripletClassifier(cfg)
    if name == "pair-state":
        return PairStateClassifier(cfg, use_slots=False, use_local_conv=False)
    if name == "pair-slot":
        return PairStateClassifier(cfg, use_slots=True, use_local_conv=False)
    if name == "hybrid-pair":
        return PairStateClassifier(cfg, use_slots=False, use_local_conv=True)
    if name == "typed-latent":
        return TypedLatentClassifier(cfg)
    raise ValueError(f"Unknown higher-order model: {name}")
