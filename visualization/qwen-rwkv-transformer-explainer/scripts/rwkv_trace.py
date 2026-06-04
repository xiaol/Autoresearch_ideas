#!/usr/bin/env python3
"""Emit Transformer Explainer-compatible JSON traces for RWKV-7 state models."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MANIFOLD_SRC = Path("/home/xiaol/X/rwkv-manifold-steering/src")
if DEFAULT_MANIFOLD_SRC.exists() and str(DEFAULT_MANIFOLD_SRC) not in sys.path:
    sys.path.insert(0, str(DEFAULT_MANIFOLD_SRC))

DEFAULT_MODEL_PATH = os.environ.get(
    "RWKV_MODEL_PATH",
    "/home/xiaol/X/models/rwkv7-g1/rwkv7a-g1d-0.1b-20260212-ctx8192.pth",
)
MODEL_KEY = "rwkv7-g1-0.1b"


@dataclass
class LoadedRwkv:
    tokenizer: Any
    model: Any
    torch: Any
    device: str
    model_path: str


def load_rwkv(model_path: str = DEFAULT_MODEL_PATH) -> LoadedRwkv:
    try:
        import torch
        from rwkv_manifold_steering.rwkv7a_model import RWKV7AModel, resolve_model_path
        from rwkv_manifold_steering.tokenizer import RWKVTokenizer
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "rwkv_trace.py needs torch and the local rwkv_manifold_steering package. "
            "Run with `PYTHONPATH=/home/xiaol/X/rwkv-manifold-steering/src uv run --with torch ...`."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("The local RWKV-7 wrapper requires CUDA, but torch cannot see CUDA.")

    resolved_model_path = str(resolve_model_path(model_path))
    tokenizer = RWKVTokenizer()
    model = RWKV7AModel(resolved_model_path, device="cuda", compile_cuda=True)
    model.eval()

    return LoadedRwkv(
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        device="cuda",
        model_path=resolved_model_path,
    )


def _decode_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)])
    return (
        text.replace("\n", "[NEWLINE]")
        .replace("\t", "[TAB]")
        .replace("\r", "[CR]")
    )


def _finite_or_none(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def _model_meta(model: Any) -> dict[str, int]:
    return {
        "layer_num": int(model.n_layer),
        "attention_head_num": int(model.n_head),
        "dimension": int(model.n_embd),
        "vocab_size": int(model.vocab_size),
    }


def _softmax(values: Any, torch: Any) -> Any:
    return torch.softmax(values, dim=0)


def _probability_payload(
    logits: Any,
    tokenizer: Any,
    torch: Any,
    temperature: float,
    sampling_type: str,
    sampling_value: float,
    selection_strategy: str,
    top_n: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scaled = logits.float() / temperature
    sorted_scaled, sorted_ids = torch.sort(scaled, descending=True)
    display_n = min(top_n, int(sorted_scaled.numel()))

    probabilities = torch.zeros_like(sorted_scaled)
    top_k_logit: list[float | None] = [None] * display_n
    top_p_probability: list[float | None] = [None] * display_n
    cumulative_probability: list[float | None] = [None] * display_n
    cutoff_index: int | None = None

    if sampling_type == "top-p":
        original_probs = _softmax(sorted_scaled, torch)
        cumulative = torch.cumsum(original_probs, dim=0)
        cutoff_matches = torch.nonzero(cumulative >= float(sampling_value), as_tuple=False)
        cutoff_index = (
            int(cutoff_matches[0].item())
            if cutoff_matches.numel()
            else int(sorted_scaled.numel() - 1)
        )
        allowed = sorted_scaled[: cutoff_index + 1]
        probabilities[: cutoff_index + 1] = _softmax(allowed, torch)
        for idx in range(display_n):
            top_p_probability[idx] = float(original_probs[idx].item())
            cumulative_probability[idx] = float(cumulative[idx].item())
    else:
        k = max(1, int(sampling_value))
        k = min(k, int(sorted_scaled.numel()))
        allowed = sorted_scaled[:k]
        probabilities[:k] = _softmax(allowed, torch)
        for idx in range(min(display_n, k)):
            top_k_logit[idx] = float(sorted_scaled[idx].item())

    if selection_strategy == "greedy":
        sampled_sorted_idx = 0
    elif seed >= 0:
        generator = torch.Generator(device=probabilities.device)
        generator.manual_seed(seed)
        sampled_sorted_idx = int(torch.multinomial(probabilities, 1, generator=generator).item())
    else:
        sampled_sorted_idx = int(torch.multinomial(probabilities, 1).item())

    max_for_exp = sorted_scaled[0]
    rows: list[dict[str, Any]] = []
    for rank in range(display_n):
        token_id = int(sorted_ids[rank].item())
        probability = float(probabilities[rank].item())
        scaled_logit = float(sorted_scaled[rank].item())
        row = {
            "rank": rank,
            "tokenId": token_id,
            "token": _decode_token(tokenizer, token_id),
            "logit": float(logits[token_id].float().item()),
            "scaledLogit": scaled_logit,
            "expLogit": float(torch.exp(sorted_scaled[rank] - max_for_exp).item()),
            "probability": probability,
            "topKLogit": top_k_logit[rank],
        }
        if sampling_type == "top-p":
            row.update(
                {
                    "topPProbability": top_p_probability[rank],
                    "cumulativeProbability": cumulative_probability[rank],
                    "cutoffIndex": cutoff_index,
                }
            )
        rows.append(row)

    if sampled_sorted_idx >= display_n:
        token_id = int(sorted_ids[sampled_sorted_idx].item())
        sampled = {
            "rank": sampled_sorted_idx,
            "tokenId": token_id,
            "token": _decode_token(tokenizer, token_id),
            "logit": float(logits[token_id].float().item()),
            "scaledLogit": float(sorted_scaled[sampled_sorted_idx].item()),
            "expLogit": float(torch.exp(sorted_scaled[sampled_sorted_idx] - max_for_exp).item()),
            "probability": float(probabilities[sampled_sorted_idx].item()),
            "topKLogit": (
                float(sorted_scaled[sampled_sorted_idx].item()) if sampling_type != "top-p" else None
            ),
        }
    else:
        sampled = rows[sampled_sorted_idx]

    return rows, sampled


def _trace_recurrent_state(
    loaded: LoadedRwkv, token_ids: list[int]
) -> tuple[Any, list[list[Any]], dict[str, Any], list[dict[str, Any]]]:
    torch = loaded.torch
    from torch.nn import functional as F
    from rwkv_manifold_steering.rwkv7a_model import rwkv7_op

    model = loaded.model
    z = model.z
    state = model.initial_state()
    previous_attention_states: list[Any | None] = [None] * model.n_layer
    hidden_by_position: list[list[Any]] = []
    layer_summaries: list[dict[str, Any]] = [
        {"layer": layer, "hiddenNorms": [], "stateNorms": [], "stateDeltas": []}
        for layer in range(model.n_layer)
    ]
    rwkv_blocks: list[dict[str, Any]] = [
        {
            "layer": layer,
            "timeMix": {
                "receptanceNorms": [],
                "decayMeans": [],
                "keyNorms": [],
                "replacementKeyNorms": [],
                "removalKeyNorms": [],
                "valueNorms": [],
                "valueResidualGateMeans": [],
                "writeGateMeans": [],
                "gateNorms": [],
                "stateBeforeNorms": [],
                "decayedStateNorms": [],
                "writeNorms": [],
                "eraseNorms": [],
                "stateAfterNorms": [],
                "readNorms": [],
                "bonusNorms": [],
                "outputNorms": [],
            },
            "state": {
                "hiddenNorms": [],
                "norms": [],
                "deltas": [],
                "rms": [],
                "stableRanks": [],
            },
            "stateTransition": {
                "stateBefore": [],
                "decayed": [],
                "erase": [],
                "write": [],
                "stateAfter": [],
            },
            "channelMix": {
                "previousStateNorms": [],
                "keyActivationNorms": [],
                "outputNorms": [],
            },
        }
        for layer in range(model.n_layer)
    ]
    logits = None

    def head_norms(tensor: Any) -> list[float]:
        values = tensor.detach().float().view(-1, model.n_head, model.head_size)[-1]
        return [float(value) for value in torch.linalg.vector_norm(values, dim=1).cpu().tolist()]

    def head_means(tensor: Any) -> list[float]:
        values = tensor.detach().float().view(-1, model.n_head, model.head_size)[-1]
        return [float(value) for value in values.mean(dim=1).cpu().tolist()]

    def state_matrix_norms(matrices: Any) -> list[float]:
        values = matrices.detach().float().reshape(model.n_head, -1)
        return [float(value) for value in torch.linalg.vector_norm(values, dim=1).cpu().tolist()]

    def downsample_matrix(matrix: Any, bins: int = 4) -> list[list[float]]:
        matrix = matrix.detach().float().abs()
        if matrix.shape[-1] % bins == 0 and matrix.shape[-2] % bins == 0:
            row_bin = matrix.shape[-2] // bins
            col_bin = matrix.shape[-1] // bins
            small = matrix.view(bins, row_bin, bins, col_bin).mean(dim=(1, 3))
        else:
            small = F.adaptive_avg_pool2d(matrix.view(1, 1, *matrix.shape[-2:]), (bins, bins))[0, 0]
        return [[float(value) for value in row] for row in small.cpu().tolist()]

    def state_matrix_payload(matrices: Any) -> list[list[list[float]]]:
        return [downsample_matrix(matrix) for matrix in matrices.detach().float()]

    def state_update_component_norms(
        state_before: Any,
        decay: Any,
        replacement_key: Any,
        removal_key: Any,
        value: Any,
        write_gate: Any,
    ) -> dict[str, list[float]]:
        previous = state_before.detach().float()
        decay_by_head = decay.detach().float().view(-1, model.n_head, model.head_size)[-1]
        write_key_by_head = replacement_key.detach().float().view(-1, model.n_head, model.head_size)[-1]
        removal_by_head = removal_key.detach().float().view(-1, model.n_head, model.head_size)[-1]
        value_by_head = value.detach().float().view(-1, model.n_head, model.head_size)[-1]
        write_gate_by_head = write_gate.detach().float().view(-1, model.n_head, model.head_size)[-1]

        decayed = previous * decay_by_head[:, None, :]
        write = value_by_head[:, :, None] * write_key_by_head[:, None, :]
        erase_vector = torch.einsum("hij,hj->hi", previous, removal_by_head)
        erase = erase_vector[:, :, None] * (removal_by_head * write_gate_by_head)[:, None, :]

        return {
            "stateBeforeNorms": state_matrix_norms(previous),
            "decayedStateNorms": state_matrix_norms(decayed),
            "writeNorms": state_matrix_norms(write),
            "eraseNorms": state_matrix_norms(erase),
            "transition": {
                "stateBefore": state_matrix_payload(previous),
                "decayed": state_matrix_payload(decayed),
                "erase": state_matrix_payload(erase),
                "write": state_matrix_payload(write),
            },
        }

    def state_quality(att_state: Any) -> tuple[list[float], list[float]]:
        matrices = att_state.detach().float()
        rms = torch.sqrt(torch.mean(matrices * matrices, dim=(1, 2)))
        stable_ranks: list[float] = []
        for matrix in matrices:
            fro = torch.linalg.matrix_norm(matrix, ord="fro")
            spectral = torch.linalg.matrix_norm(matrix, ord=2)
            value = (fro / torch.clamp(spectral, min=1e-8)) ** 2
            stable_ranks.append(float(value.item()))
        return [float(value) for value in rms.cpu().tolist()], stable_ranks

    def time_mix_with_metrics(
        layer_id: int,
        x: Any,
        x_prev: Any,
        v_first: Any,
        recurrent_state: Any,
    ) -> tuple[Any, Any, Any, Any, dict[str, list[float]]]:
        block = f"blocks.{layer_id}."
        att = f"{block}att."
        time_steps = x.shape[0]
        xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
        xr, xw, xk, xv, xa, xg = (
            x + xx * z[f"{att}x_r"],
            x + xx * z[f"{att}x_w"],
            x + xx * z[f"{att}x_k"],
            x + xx * z[f"{att}x_v"],
            x + xx * z[f"{att}x_a"],
            x + xx * z[f"{att}x_g"],
        )

        r = (xr @ z[f"{att}receptance.weight"]).contiguous()
        w = (torch.tanh(xw @ z[f"{att}w1"]) @ z[f"{att}w2"]).contiguous()
        k = (xk @ z[f"{att}key.weight"]).contiguous()
        v = (xv @ z[f"{att}value.weight"]).contiguous()
        write_gate = torch.sigmoid(z[f"{att}a0"] + (xa @ z[f"{att}a1"]) @ z[f"{att}a2"]).contiguous()
        value_residual_gate = torch.sigmoid(
            z[f"{att}v0"] + (xv @ z[f"{att}v1"]) @ z[f"{att}v2"]
        ).contiguous()
        value_residual_gate_for_metrics = value_residual_gate
        g = (torch.sigmoid(xg @ z[f"{att}g1"]) @ z[f"{att}g2"]).contiguous()

        removal_key = k * z[f"{att}k_k"]
        kk = F.normalize(removal_key.view(time_steps, model.n_head, model.head_size), dim=-1, p=2.0).view(
            time_steps, model.n_head * model.head_size
        )
        raw_key = k
        k = (k * (1 + (write_gate - 1) * z[f"{att}k_a"])).contiguous()
        if layer_id == 0:
            v_first = v
            value_residual_gate_for_metrics = torch.zeros_like(value_residual_gate)
        else:
            v = (v + (v_first - v) * value_residual_gate).contiguous()

        w = (-F.softplus(-(z[f"{att}w0"] + w)) - 0.5).contiguous()
        decay = torch.exp(-torch.exp(w.detach().float()))
        state_before = recurrent_state.detach().float().clone()
        update_components = state_update_component_norms(
            state_before=state_before,
            decay=decay,
            replacement_key=k,
            removal_key=kk,
            value=v,
            write_gate=write_gate,
        )
        state_out = rwkv7_op(recurrent_state, r, w, k, v, (-kk).contiguous(), (kk * write_gate).contiguous())
        state_after = recurrent_state.detach().float().clone()
        read_pre_norms = head_norms(state_out)

        state_out = F.group_norm(
            state_out.view(time_steps, model.n_head * model.head_size),
            num_groups=model.n_head,
            weight=z[f"{att}ln_x.weight"],
            bias=z[f"{att}ln_x.bias"],
            eps=64e-5,
        ).view(time_steps, model.n_head * model.head_size)
        bonus = (
            (r * k * z[f"{att}r_k"]).view(time_steps, model.n_head, model.head_size).sum(dim=-1, keepdim=True)
            * v.view(time_steps, model.n_head, model.head_size)
        ).view(time_steps, model.n_head * model.head_size)
        state_out = state_out + bonus
        mixed = state_out * g
        output = mixed @ z[f"{att}output.weight"]
        metrics = {
            "receptanceNorms": head_norms(r),
            "decayMeans": head_means(decay),
            "keyNorms": head_norms(raw_key),
            "replacementKeyNorms": head_norms(k),
            "removalKeyNorms": head_norms(removal_key),
            "valueNorms": head_norms(v),
            "valueResidualGateMeans": head_means(value_residual_gate_for_metrics),
            "writeGateMeans": head_means(write_gate),
            "gateNorms": head_norms(g),
            "stateBeforeNorms": update_components["stateBeforeNorms"],
            "decayedStateNorms": update_components["decayedStateNorms"],
            "writeNorms": update_components["writeNorms"],
            "eraseNorms": update_components["eraseNorms"],
            "stateAfterNorms": state_matrix_norms(state_after),
            "transition": {
                **update_components["transition"],
                "stateAfter": state_matrix_payload(state_after),
            },
            "readNorms": read_pre_norms,
            "bonusNorms": head_norms(bonus),
            "outputNorms": head_norms(mixed),
        }
        return output, x[-1, :], recurrent_state, v_first, metrics

    def channel_mix_with_metrics(layer_id: int, x: Any, x_prev: Any, token_id: int) -> tuple[Any, Any, dict[str, float]]:
        ffn = f"blocks.{layer_id}.ffn."
        time_steps, _ = x.shape
        xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
        k = x + xx * z[f"{ffn}x_k"]
        k = torch.relu(k @ z[f"{ffn}key.weight"]) ** 2
        ss = (x @ z[f"{ffn}s1"]).view(time_steps, 1, 32) @ z[f"{ffn}s_emb.weight"][token_id].view(time_steps, 32, 32)
        k = k * ((ss.view(time_steps, 32) @ z[f"{ffn}s2"]) + z[f"{ffn}s0"])
        output = k @ z[f"{ffn}value.weight"]
        metrics = {
            "previousStateNorms": float(torch.linalg.vector_norm(x_prev.detach().float()).item()),
            "keyActivationNorms": float(torch.linalg.vector_norm(k[-1].detach().float()).item()),
            "outputNorms": float(torch.linalg.vector_norm(output[-1].detach().float()).item()),
        }
        return output, x[-1, :], metrics

    with torch.no_grad():
        for token_id in token_ids:
            idx = [int(token_id)]
            x = z["emb.weight"][idx].contiguous()
            v_first = torch.empty_like(x)
            hidden_by_layer = []

            for layer in range(model.n_layer):
                block = f"blocks.{layer}."
                xx = F.layer_norm(
                    x,
                    (model.n_embd,),
                    weight=z[f"{block}ln1.weight"],
                    bias=z[f"{block}ln1.bias"],
                ).contiguous()

                xx, state[layer * 3 + 0], state[layer * 3 + 1], v_first, time_metrics = time_mix_with_metrics(
                    layer,
                    xx,
                    state[layer * 3 + 0],
                    v_first,
                    state[layer * 3 + 1],
                )
                x = x + xx

                xx = F.layer_norm(
                    x,
                    (model.n_embd,),
                    weight=z[f"{block}ln2.weight"],
                    bias=z[f"{block}ln2.bias"],
                ).contiguous()

                xx, state[layer * 3 + 2], channel_metrics = channel_mix_with_metrics(
                    layer,
                    xx,
                    state[layer * 3 + 2],
                    int(token_id),
                )
                x = x + xx
                hidden_by_layer.append(x[-1, :].detach().float().cpu())

                transition_metrics = time_metrics.pop("transition")
                for key, value in time_metrics.items():
                    rwkv_blocks[layer]["timeMix"][key].append(value)
                for key, value in transition_metrics.items():
                    rwkv_blocks[layer]["stateTransition"][key].append(value)
                for key, value in channel_metrics.items():
                    rwkv_blocks[layer]["channelMix"][key].append(value)

            logits = (
                F.layer_norm(x[-1, :], (model.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
                @ z["head.weight"]
            ).detach().float().cpu()

            hidden_by_position.append([hidden.detach().float().cpu() for hidden in hidden_by_layer])

            for layer in range(model.n_layer):
                hidden = hidden_by_layer[layer].detach().float()
                att_state = state[layer * 3 + 1].detach().float()
                state_by_head = att_state.reshape(model.n_head, -1)
                state_norms = torch.linalg.vector_norm(state_by_head, dim=1)
                state_rms, stable_ranks = state_quality(att_state)

                previous = previous_attention_states[layer]
                if previous is None:
                    deltas = state_norms
                else:
                    delta_by_head = (att_state - previous).reshape(model.n_head, -1)
                    deltas = torch.linalg.vector_norm(delta_by_head, dim=1)

                layer_summaries[layer]["hiddenNorms"].append(
                    float(torch.linalg.vector_norm(hidden).item())
                )
                layer_summaries[layer]["stateNorms"].append(
                    [float(value) for value in state_norms.detach().cpu().tolist()]
                )
                layer_summaries[layer]["stateDeltas"].append(
                    [float(value) for value in deltas.detach().cpu().tolist()]
                )
                rwkv_blocks[layer]["state"]["hiddenNorms"].append(
                    float(torch.linalg.vector_norm(hidden).item())
                )
                rwkv_blocks[layer]["state"]["norms"].append(
                    [float(value) for value in state_norms.detach().cpu().tolist()]
                )
                rwkv_blocks[layer]["state"]["deltas"].append(
                    [float(value) for value in deltas.detach().cpu().tolist()]
                )
                rwkv_blocks[layer]["state"]["rms"].append(state_rms)
                rwkv_blocks[layer]["state"]["stableRanks"].append(stable_ranks)
                previous_attention_states[layer] = att_state.clone()

    if logits is None:
        raise ValueError("token_ids cannot be empty")

    rwkv_state = {
        "type": "rwkv7-recurrent-state",
        "headSize": int(model.head_size),
        "description": (
            "Per-token RWKV recurrent attention-state norms and deltas. Matrix slots in this "
            "trace contain normalized causal hidden-state similarity, not Transformer QK attention."
        ),
        "layers": layer_summaries,
    }
    return logits, hidden_by_position, rwkv_state, rwkv_blocks


def _state_similarity_outputs(
    hidden_by_position: list[list[Any]], meta: dict[str, int], torch: Any
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    seq_len = len(hidden_by_position)
    if seq_len == 0:
        return outputs

    layer_num = meta["layer_num"]
    heads = meta["attention_head_num"]
    head_size = meta["dimension"] // heads

    for layer in range(layer_num):
        hiddens = torch.stack([hidden_by_position[pos][layer] for pos in range(seq_len)], dim=0)
        for head in range(heads):
            start = head * head_size
            end = start + head_size
            head_vectors = hiddens[:, start:end].float()
            head_vectors = torch.nn.functional.normalize(head_vectors, dim=1, eps=1e-6)
            similarity = torch.matmul(head_vectors, head_vectors.t())

            clean_rows: list[list[float]] = []
            masked_rows: list[list[float | None]] = []
            for row_idx in range(seq_len):
                causal_values = torch.relu(similarity[row_idx, : row_idx + 1]) + 1e-6
                causal_values = causal_values / causal_values.sum()
                clean_row = [0.0] * seq_len
                masked_row: list[float | None] = [None] * seq_len
                for col_idx, value in enumerate(causal_values.tolist()):
                    clean_value = _finite_or_none(float(value)) or 0.0
                    clean_row[col_idx] = clean_value
                    masked_row[col_idx] = clean_value
                clean_rows.append(clean_row)
                masked_rows.append(masked_row)

            dims = [seq_len, seq_len]
            size = seq_len * seq_len
            for suffix, data in (
                ("attn", clean_rows),
                ("attn_scaled", clean_rows),
                ("attn_masked", masked_rows),
                ("attn_softmax", clean_rows),
                ("attn_dropout", clean_rows),
            ):
                outputs[f"block_{layer}_attn_head_{head}_{suffix}"] = {
                    "data": data,
                    "dims": dims,
                    "size": size,
                }

    return outputs


def build_trace(
    loaded: LoadedRwkv,
    input_text: str,
    temperature: float = 0.8,
    sampling_type: str = "top-k",
    sampling_value: float = 5,
    selection_strategy: str = "greedy",
    top_n: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    tokenizer = loaded.tokenizer
    torch = loaded.torch
    token_ids = tokenizer.encode(input_text or " ")
    if not token_ids:
        token_ids = tokenizer.encode(" ")

    tokens = [_decode_token(tokenizer, token_id) for token_id in token_ids]
    logits, hidden_by_position, rwkv_state, rwkv_blocks = _trace_recurrent_state(loaded, token_ids)
    meta = _model_meta(loaded.model)
    probabilities, sampled = _probability_payload(
        logits=logits,
        tokenizer=tokenizer,
        torch=torch,
        temperature=temperature,
        sampling_type=sampling_type,
        sampling_value=sampling_value,
        selection_strategy=selection_strategy,
        top_n=top_n,
        seed=seed,
    )
    outputs = _state_similarity_outputs(hidden_by_position, meta, torch)
    note = (
        "RWKV-7 trace mode uses a recurrent state model. The matrix view shows normalized "
        "causal hidden-state similarity by layer/head, not Transformer QK attention weights."
    )

    return {
        "prompt": input_text,
        "modelId": loaded.model_path,
        "modelKey": MODEL_KEY,
        "modelMeta": meta,
        "tokens": tokens,
        "tokenIds": token_ids,
        "logits": [],
        "outputs": outputs,
        "probabilities": probabilities,
        "sampled": sampled,
        "rwkvState": rwkv_state,
        "rwkvBlocks": rwkv_blocks,
        "note": note,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Data visualization empowers users to")
    parser.add_argument("--model-path", "--model-id", dest="model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sampling-type", choices=["top-k", "top-p"], default="top-k")
    parser.add_argument("--sampling-value", type=float, default=5)
    parser.add_argument("--selection-strategy", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_rwkv(args.model_path)
    trace = build_trace(
        loaded=loaded,
        input_text=args.input,
        temperature=args.temperature,
        sampling_type=args.sampling_type,
        sampling_value=args.sampling_value,
        selection_strategy=args.selection_strategy,
        top_n=args.top_n,
        seed=args.seed,
    )
    json.dump(trace, sys.stdout, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
