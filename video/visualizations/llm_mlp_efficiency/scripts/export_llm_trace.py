#!/usr/bin/env python3
"""Export compact LLM telemetry for the browser visualizer.

The browser should never receive full 0.8B model weights. This script samples
hidden-state and generation summaries into a small JSON trace.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an LLM trace JSON for the Three.js visualizer.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="Hugging Face model id or local path.")
    parser.add_argument("--prompt", required=True, help="Prompt to trace.")
    parser.add_argument("--out", default="public/traces/qwen35-08b-real.json", help="Output JSON path.")
    parser.add_argument("--max-new-tokens", type=int, default=6, help="Generated tokens to include.")
    parser.add_argument("--max-input-tokens", type=int, default=128, help="Maximum prompt tokens before generation.")
    parser.add_argument("--top-k", type=int, default=6, help="Top candidates stored per generated token.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Use 0 for greedy decoding.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Execution device.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Model dtype.")
    parser.add_argument("--device-map", default=None, help="Optional Transformers device_map, e.g. auto.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to Transformers.")
    parser.add_argument("--attentions", action="store_true", help="Request attention tensors when supported.")
    parser.add_argument("--chat-template", action="store_true", help="Wrap prompt with tokenizer chat template.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:  # pragma: no cover - depends on installed transformers version.
            AutoModelForImageTextToText = None
    except ImportError as error:
        print(f"Missing dependency: {error}. Install scripts/requirements.txt first.", file=sys.stderr)
        return 2

    trust_remote_code = bool(args.trust_remote_code)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trust_remote_code)
    model = load_model(args, torch, AutoModelForCausalLM, AutoModelForImageTextToText)
    model.eval()

    prompt_text = args.prompt
    if args.chat_template and hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    model_device = infer_model_device(model, torch)
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    input_length = int(encoded["input_ids"].shape[-1])

    generate_kwargs: dict[str, Any] = {
        **encoded,
        "max_new_tokens": args.max_new_tokens,
        "return_dict_in_generate": True,
        "output_scores": True,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
    if tokenizer.eos_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

    with torch.no_grad():
        generation = model.generate(**generate_kwargs)

    sequence = generation.sequences[0]
    attention_mask = torch.ones_like(sequence, dtype=torch.long, device=sequence.device).unsqueeze(0)
    forward_kwargs = {
        "input_ids": sequence.unsqueeze(0),
        "attention_mask": attention_mask,
        "output_hidden_states": True,
        "output_attentions": bool(args.attentions),
        "use_cache": False,
        "return_dict": True,
    }
    with torch.no_grad():
        outputs = model(**forward_kwargs)

    hidden_states = list(getattr(outputs, "hidden_states", []) or [])
    if len(hidden_states) < 2:
        raise RuntimeError("Model did not return hidden_states. Check model support for output_hidden_states=True.")

    attentions = list(getattr(outputs, "attentions", []) or [])
    text_config = getattr(config, "text_config", config)
    layers = build_layers(text_config, len(hidden_states) - 1)
    top_tokens = decode_top_tokens(generation.scores, tokenizer, args.top_k, torch)
    tokens = decode_sequence_tokens(sequence, tokenizer, input_length)
    steps = build_steps(
        layers=layers,
        hidden_states=hidden_states,
        attentions=attentions,
        input_length=input_length,
        tokens=tokens,
        top_tokens=top_tokens,
        tokenizer=tokenizer,
        torch=torch,
    )

    trace = {
        "schemaVersion": "llm_trace_v1",
        "traceKind": "real_hf_forward",
        "model": {
            "name": args.model,
            "family": infer_model_family(getattr(config, "model_type", None)),
            "modelType": getattr(config, "model_type", None),
            "architecture": first_or_none(getattr(config, "architectures", None)),
            "hiddenSize": int_or_none(getattr(text_config, "hidden_size", None)),
            "contextLength": int_or_none(getattr(text_config, "max_position_embeddings", None)),
            "vocabSize": int_or_none(getattr(text_config, "vocab_size", None)),
        },
        "architecture": {
            "numLayers": len(layers),
            "hiddenSize": int_or_none(getattr(text_config, "hidden_size", None)),
            "contextLength": int_or_none(getattr(text_config, "max_position_embeddings", None)),
            "intermediateSize": int_or_none(getattr(text_config, "intermediate_size", None)),
            "blockKind": infer_block_kind(getattr(config, "model_type", None)),
            "layers": layers,
        },
        "prompt": args.prompt,
        "generatedText": tokenizer.decode(sequence[input_length:], skip_special_tokens=False),
        "tokens": tokens,
        "steps": steps,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with {len(tokens)} tokens, {len(layers)} layers, {len(steps)} steps.")
    return 0


def load_model(args: argparse.Namespace, torch: Any, causal_cls: Any, image_text_cls: Any) -> Any:
    dtype = choose_dtype(args, torch)
    kwargs: dict[str, Any] = {"trust_remote_code": bool(args.trust_remote_code)}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if args.device_map:
        kwargs["device_map"] = args.device_map

    errors: list[str] = []
    for model_cls in [causal_cls, image_text_cls]:
        if model_cls is None:
            continue
        try:
            model = model_cls.from_pretrained(args.model, **kwargs)
            if not args.device_map:
                model.to(resolve_device(args.device, torch))
            return model
        except Exception as error:  # pragma: no cover - depends on model package support.
            errors.append(f"{model_cls.__name__}: {error}")

    joined = "\n".join(errors)
    raise RuntimeError(
        "Could not load the model with available Transformers auto classes.\n"
        "For Qwen3.5, install a Transformers build that supports qwen3_5 and retry.\n"
        f"{joined}"
    )


def choose_dtype(args: argparse.Namespace, torch: Any) -> Any:
    if args.dtype == "float32":
        return torch.float32
    if args.dtype == "float16":
        return torch.float16
    if args.dtype == "bfloat16":
        return torch.bfloat16
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        return torch.bfloat16
    return torch.float32


def resolve_device(device: str, torch: Any) -> Any:
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if device == "mps" or (device == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def infer_model_device(model: Any, torch: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def build_layers(text_config: Any, observed_layers: int) -> list[dict[str, Any]]:
    layer_types = list(getattr(text_config, "layer_types", []) or [])
    layer_count = int(getattr(text_config, "num_hidden_layers", observed_layers) or observed_layers)
    hidden_size = int_or_none(getattr(text_config, "hidden_size", None))
    intermediate_size = int_or_none(getattr(text_config, "intermediate_size", None))
    model_type = str(getattr(text_config, "model_type", "") or "").lower()
    is_rwkv = "rwkv" in model_type
    head_dim = int_or_none(getattr(text_config, "head_dim", None))
    full_interval = int(getattr(text_config, "full_attention_interval", 4) or 4)
    layers: list[dict[str, Any]] = []
    for index in range(layer_count):
        if is_rwkv:
            kind = "rwkv7_block"
            heads = int(hidden_size / head_dim) if hidden_size and head_dim else int(getattr(text_config, "num_heads", 12) or 12)
        else:
            kind = layer_types[index] if index < len(layer_types) else ("full_attention" if (index + 1) % full_interval == 0 else "linear_attention")
            heads = int(getattr(text_config, "num_attention_heads", 8) if kind == "full_attention" else getattr(text_config, "linear_num_key_heads", 16))
        layers.append(
            {
                "index": index,
                "name": f"Block {index + 1:02d}",
                "kind": kind,
                "heads": heads,
                "headDim": head_dim,
                "hiddenSize": hidden_size,
                "intermediateSize": intermediate_size,
            }
        )
    return layers


def decode_top_tokens(scores: Any, tokenizer: Any, top_k: int, torch: Any) -> list[list[dict[str, Any]]]:
    decoded: list[list[dict[str, Any]]] = []
    for score in list(scores or []):
        probabilities = torch.softmax(score[0].detach().float().cpu(), dim=-1)
        values, indices = torch.topk(probabilities, min(top_k, probabilities.numel()))
        decoded.append(
            [
                {
                    "token": display_token(tokenizer.decode([int(token_id)], skip_special_tokens=False)),
                    "probability": float(probability),
                }
                for probability, token_id in zip(values.tolist(), indices.tolist())
            ]
        )
    return decoded


def decode_sequence_tokens(sequence: Any, tokenizer: Any, input_length: int) -> list[dict[str, Any]]:
    token_ids = [int(token_id) for token_id in sequence.detach().cpu().tolist()]
    return [
        {
            "index": index,
            "id": token_id,
            "text": display_token(tokenizer.decode([token_id], skip_special_tokens=False)),
            "source": "prompt" if index < input_length else "generated",
        }
        for index, token_id in enumerate(token_ids)
    ]


def build_steps(
    layers: list[dict[str, Any]],
    hidden_states: list[Any],
    attentions: list[Any],
    input_length: int,
    tokens: list[dict[str, Any]],
    top_tokens: list[list[dict[str, Any]]],
    tokenizer: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    layer_count = min(len(layers), len(hidden_states) - 1)
    generated_count = max(1, len(tokens) - input_length)
    steps: list[dict[str, Any]] = []
    for step_index in range(generated_count + 1):
        active_pos = input_length - 1 if step_index == 0 else min(input_length + step_index - 1, len(tokens) - 1)
        raw_metrics = [
            layer_metric(
                layer_index=layer_index,
                layer=layers[layer_index],
                hidden_states=hidden_states,
                attentions=attentions,
                active_pos=active_pos,
                torch=torch,
            )
            for layer_index in range(layer_count)
        ]
        metrics = normalize_metrics(raw_metrics)
        generated_token = "" if step_index == 0 or active_pos >= len(tokens) else tokens[active_pos]["text"]
        steps.append(
            {
                "index": step_index,
                "phase": "prompt" if step_index == 0 else "decode",
                "label": "Prompt pass" if step_index == 0 else f"Decode {step_index}",
                "activeTokenIndex": active_pos,
                "generatedToken": generated_token,
                "topTokens": top_tokens[min(step_index, len(top_tokens) - 1)] if top_tokens else [],
                "layers": metrics,
            }
        )
    return steps


def layer_metric(layer_index: int, layer: dict[str, Any], hidden_states: list[Any], attentions: list[Any], active_pos: int, torch: Any) -> dict[str, Any]:
    before = hidden_states[layer_index][0, active_pos].detach().float().cpu()
    after = hidden_states[layer_index + 1][0, active_pos].detach().float().cpu()
    hidden_dim = max(1, int(after.numel()))
    residual_norm = float(after.norm().item() / math.sqrt(hidden_dim))
    delta_norm = float((after - before).norm().item() / math.sqrt(hidden_dim))
    attention_strength = delta_norm
    entropy = 0.0
    heads = synthetic_heads(layer, layer_index, delta_norm)

    if layer_index < len(attentions) and attentions[layer_index] is not None:
        attn = attentions[layer_index]
        if hasattr(attn, "detach") and attn.ndim == 4 and active_pos < attn.shape[-2]:
            focus = attn[0, :, active_pos, : active_pos + 1].detach().float().cpu().clamp_min(1e-9)
            entropy_by_head = -(focus * focus.log()).sum(dim=-1) / math.log(max(2, focus.shape[-1]))
            entropy = float(entropy_by_head.mean().item())
            head_values = (1.0 - entropy_by_head).clamp(0, 1).tolist()
            heads = [
                {"index": index, "label": f"H{index + 1}", "value": float(value)}
                for index, value in enumerate(head_values[:6])
            ]
            attention_strength = float((1.0 - entropy_by_head.mean()).clamp(0, 1).item())

    return {
        "layerIndex": layer_index,
        "attention": attention_strength,
        "mlp": delta_norm,
        "residualNorm": residual_norm,
        "entropy": entropy,
        "heads": heads,
        "note": layer_note(layer),
        "efficiency": mlp_efficiency_metric(layer, delta_norm),
    }


def normalize_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_residual = max([metric["residualNorm"] for metric in metrics] + [1e-6])
    max_mlp = max([metric["mlp"] for metric in metrics] + [1e-6])
    max_attention = max([metric["attention"] for metric in metrics] + [1e-6])
    normalized = []
    for metric in metrics:
        normalized_metric = {
            **metric,
            "residualNorm": clamp(metric["residualNorm"] / max_residual),
            "mlp": clamp(metric["mlp"] / max_mlp),
            "attention": clamp(metric["attention"] / max_attention),
            "entropy": clamp(metric.get("entropy", 0.0)),
        }
        efficiency = dict(metric.get("efficiency") or {})
        if efficiency and efficiency.get("estimated", True):
            activity = normalized_metric["mlp"]
            residual = normalized_metric["residualNorm"]
            intermediate_size = max(1, int_or_none(metric.get("intermediateSize")) or int_or_none(efficiency.get("intermediateSize")) or 0)
            active_fraction = clamp(0.06 + activity * 0.34)
            efficiency["activeFraction"] = active_fraction
            efficiency["activeNeurons"] = round(active_fraction * intermediate_size)
            efficiency["topKCoverage"] = clamp(0.42 + activity * 0.34 + residual * 0.1)
        normalized_metric["efficiency"] = efficiency
        normalized.append(normalized_metric)
    return normalized


def mlp_efficiency_metric(layer: dict[str, Any], delta_norm: float) -> dict[str, Any]:
    flops_per_token = estimate_mlp_flops(layer)
    mega_flops_per_token = flops_per_token / 1_000_000 if flops_per_token else 0.0
    hidden_size = int_or_none(layer.get("hiddenSize")) or 0
    intermediate_size = int_or_none(layer.get("intermediateSize")) or hidden_size * 4
    active_fraction = clamp(0.06 + clamp(delta_norm) * 0.34)
    return {
        "flopsPerToken": flops_per_token,
        "megaFlopsPerToken": mega_flops_per_token,
        "activeFraction": active_fraction,
        "activeNeurons": round(active_fraction * max(1, intermediate_size)),
        "intermediateSize": intermediate_size,
        "topK": 32,
        "topKCoverage": clamp(0.42 + clamp(delta_norm) * 0.4),
        "deltaPerMFlop": delta_norm / mega_flops_per_token if mega_flops_per_token else 0.0,
        "estimated": True,
    }


def estimate_mlp_flops(layer: dict[str, Any]) -> int:
    hidden_size = int_or_none(layer.get("hiddenSize")) or 0
    intermediate_size = int_or_none(layer.get("intermediateSize")) or hidden_size * 4
    if hidden_size <= 0 or intermediate_size <= 0:
        return 0
    return 6 * hidden_size * intermediate_size


def synthetic_heads(layer: dict[str, Any], layer_index: int, base: float) -> list[dict[str, Any]]:
    count = min(6, int(layer.get("heads") or 8))
    prefix = "T" if "rwkv" in str(layer.get("kind", "")).lower() else "D" if layer.get("kind") == "linear_attention" else "H"
    return [
        {
            "index": index,
            "label": prefix + str(index + 1),
            "value": clamp((math.sin((layer_index + 1) * (index + 3)) + 1.0) * 0.25 + base * 0.4),
        }
        for index in range(count)
    ]


def infer_model_family(model_type: Any) -> str | None:
    if "rwkv" in str(model_type or "").lower():
        return "RWKV-7"
    return None


def infer_block_kind(model_type: Any) -> str | None:
    if "rwkv" in str(model_type or "").lower():
        return "rwkv7_block"
    return None


def layer_note(layer: dict[str, Any]) -> str:
    kind = str(layer.get("kind", ""))
    if "rwkv" in kind.lower():
        return "RWKV recurrent time/channel mix block"
    if kind == "full_attention":
        return "full attention block"
    return "linear state block"


def display_token(token: str) -> str:
    return token.replace("\n", "\\n").replace("\t", "\\t") or "∅"


def clamp(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def first_or_none(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
