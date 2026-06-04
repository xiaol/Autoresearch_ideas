#!/usr/bin/env python3
"""Emit Transformer Explainer-compatible JSON traces for Qwen text models."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _cached_qwen35_path() -> str | None:
    root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots"
    if not root.exists():
        return None
    snapshots = sorted(root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
    for snapshot in snapshots:
        if (snapshot / "config.json").exists() and any(snapshot.glob("*.safetensors*")):
            return str(snapshot)
    return None


DEFAULT_MODEL_ID = os.environ.get("QWEN_MODEL_ID") or _cached_qwen35_path() or "Qwen/Qwen3.5-0.8B-Base"


@dataclass
class LoadedQwen:
    tokenizer: Any
    model: Any
    torch: Any
    device: str


def _model_key(model_id: str) -> str:
    if "Qwen3.5-0.8B" in model_id:
        return "qwen3.5-0.8b"
    if "Qwen3-0.6B" in model_id:
        return "qwen3-0.6b"
    return model_id.lower().replace("/", "-").replace(".", "-")


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def _model_meta(model: Any) -> dict[str, int]:
    cfg = _text_config(model.config)
    return {
        "layer_num": int(getattr(cfg, "num_hidden_layers")),
        "attention_head_num": int(getattr(cfg, "num_attention_heads")),
        "dimension": int(getattr(cfg, "hidden_size")),
        "vocab_size": int(getattr(cfg, "vocab_size")),
    }


def load_qwen(model_id: str = DEFAULT_MODEL_ID) -> LoadedQwen:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "qwen_trace.py needs torch and transformers. "
            "Run `uv pip install torch 'transformers>=4.57.0' accelerate` in .venv, "
            "or let the Svelte endpoint invoke `uv run --with ...`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: Any = "auto" if device == "cuda" else torch.float32

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="eager",
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    model.eval()
    model.to(device)
    return LoadedQwen(tokenizer=tokenizer, model=model, torch=torch, device=device)


def _decode_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return (
        text.replace("\n", "[NEWLINE]")
        .replace("\t", "[TAB]")
        .replace("\r", "[CR]")
    )


def _finite_or_none(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def _attention_layer_indices(model: Any, attentions: Any) -> list[int]:
    cfg = _text_config(model.config)
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types:
        full_attention_layers = [
            index for index, layer_type in enumerate(layer_types) if layer_type == "full_attention"
        ]
        if attentions and len(attentions) == len(full_attention_layers):
            return full_attention_layers
    return list(range(len(attentions or [])))


def _attention_outputs(
    attentions: Any, meta: dict[str, int], layer_indices: list[int] | None = None
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    if not attentions:
        return outputs

    for attention_idx, layer_attention in enumerate(attentions):
        if layer_attention is None:
            continue
        layer_idx = layer_indices[attention_idx] if layer_indices else attention_idx

        layer = layer_attention.detach().float().cpu()
        if layer.ndim != 4:
            continue

        heads = min(int(layer.shape[1]), meta["attention_head_num"])
        seq_len = int(layer.shape[-1])
        for head_idx in range(heads):
            matrix = layer[0, head_idx].tolist()
            masked = [
                [_finite_or_none(float(value)) if col_idx <= row_idx else None for col_idx, value in enumerate(row)]
                for row_idx, row in enumerate(matrix)
            ]
            clean = [[_finite_or_none(float(value)) for value in row] for row in matrix]
            dims = [seq_len, seq_len]
            size = seq_len * seq_len
            for suffix, data in (
                ("attn", clean),
                ("attn_scaled", clean),
                ("attn_masked", masked),
                ("attn_softmax", clean),
                ("attn_dropout", clean),
            ):
                outputs[f"block_{layer_idx}_attn_head_{head_idx}_{suffix}"] = {
                    "data": data,
                    "dims": dims,
                    "size": size,
                }

    return outputs


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
        cutoff_index = int(cutoff_matches[0].item()) if cutoff_matches.numel() else int(sorted_scaled.numel() - 1)
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
            "topKLogit": float(sorted_scaled[sampled_sorted_idx].item()) if sampling_type != "top-p" else None,
        }
    else:
        sampled = rows[sampled_sorted_idx]

    return rows, sampled


def build_trace(
    loaded: LoadedQwen,
    input_text: str,
    temperature: float = 0.8,
    sampling_type: str = "top-k",
    sampling_value: float = 5,
    selection_strategy: str = "greedy",
    top_n: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    torch = loaded.torch
    tokenizer = loaded.tokenizer
    model = loaded.model
    model_id = getattr(model.config, "_name_or_path", DEFAULT_MODEL_ID)

    encoded = tokenizer(input_text or " ", return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(loaded.device) for key, value in encoded.items()}

    with torch.no_grad():
        result = model(
            **encoded,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    tokens = [_decode_token(tokenizer, int(token_id)) for token_id in input_ids]
    logits = result.logits[0, -1].detach().float().cpu()
    meta = _model_meta(model)
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

    attentions = getattr(result, "attentions", None)
    outputs = _attention_outputs(attentions, meta, _attention_layer_indices(model, attentions))
    note = (
        "Qwen trace mode maps Hugging Face post-softmax attention weights into the "
        "Transformer Explainer matrix slots; it does not expose GPT-2-style raw QK internals."
    )

    return {
        "prompt": input_text,
        "modelId": model_id,
        "modelKey": _model_key(model_id),
        "modelMeta": meta,
        "tokens": tokens,
        "tokenIds": input_ids,
        "logits": [],
        "outputs": outputs,
        "probabilities": probabilities,
        "sampled": sampled,
        "note": note,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Data visualization empowers users to")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sampling-type", choices=["top-k", "top-p"], default="top-k")
    parser.add_argument("--sampling-value", type=float, default=5)
    parser.add_argument("--selection-strategy", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_qwen(args.model_id)
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
