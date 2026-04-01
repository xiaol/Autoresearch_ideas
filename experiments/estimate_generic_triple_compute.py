from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from auto_research_llm_ideas.experiments.data import BYTE_VOCAB_SIZE
from auto_research_llm_ideas.model import triple_latent_config


DEFAULT_MODELS = [
    "transformer",
    "triple-hybrid",
    "triple-hybrid-assoc-gated-last-logits",
]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def topk_mix_flops(seq_len: int, topk: int, width: int) -> int:
    total = 0
    for token_index in range(seq_len):
        memory_size = token_index
        if memory_size == 0:
            continue
        total += 2 * min(topk, memory_size) * width
    return total


def transformer_forward_flops(seq_len: int, d_model: int, n_layers: int, vocab_size: int) -> Dict[str, int]:
    qkv_proj = 6 * seq_len * d_model * d_model * n_layers
    attn_out_proj = 2 * seq_len * d_model * d_model * n_layers
    attn_scores = 2 * seq_len * seq_len * d_model * n_layers
    attn_reduce = 2 * seq_len * seq_len * d_model * n_layers
    ffn = 16 * seq_len * d_model * d_model * n_layers
    lm_head = 2 * seq_len * d_model * vocab_size
    return {
        "qkv_proj": qkv_proj,
        "attn_out_proj": attn_out_proj,
        "attn_scores": attn_scores,
        "attn_reduce": attn_reduce,
        "ffn": ffn,
        "lm_head": lm_head,
    }


def triple_hybrid_forward_flops(
    seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    vocab_size: int,
) -> Dict[str, int]:
    inner_dim = n_heads * state_dim
    recurrent_proj = (8 * seq_len * d_model * inner_dim + 4 * seq_len * d_model * n_heads) * n_layers
    recurrent_memory = seq_len * (6 * n_heads * state_dim * state_dim + n_heads * state_dim) * n_layers
    output_proj = 2 * seq_len * inner_dim * d_model * n_layers
    ffn = 16 * seq_len * d_model * d_model * n_layers
    local_conv = 10 * seq_len * d_model * d_model
    lm_head = 2 * seq_len * d_model * vocab_size
    return {
        "recurrent_proj": recurrent_proj,
        "recurrent_memory": recurrent_memory,
        "output_proj": output_proj,
        "ffn": ffn,
        "local_conv": local_conv,
        "lm_head": lm_head,
    }


def assoc_memory_flops(
    *,
    seq_len: int,
    d_model: int,
    assoc_dim: int,
    assoc_topk: int,
    layer_multiplier: int,
    use_assoc_read_gate: bool,
    use_assoc_write_gate: bool,
    logits_fusion: bool,
    vocab_size: int,
) -> Dict[str, int]:
    assoc_proj = layer_multiplier * (
        2 * seq_len * d_model * assoc_dim
        + 2 * seq_len * d_model * d_model
        + 2 * seq_len * d_model * assoc_dim
    )
    assoc_scores = layer_multiplier * assoc_dim * seq_len * (seq_len - 1)
    assoc_topk_mix = layer_multiplier * topk_mix_flops(seq_len, assoc_topk, d_model)
    read_gate = layer_multiplier * (2 * seq_len * d_model if use_assoc_read_gate else 0)
    write_gate = layer_multiplier * (4 * max(seq_len - 1, 0) * d_model if use_assoc_write_gate else 0)
    assoc_lm_head = 2 * seq_len * d_model * vocab_size if logits_fusion else 0
    return {
        "assoc_proj": assoc_proj,
        "assoc_scores": assoc_scores,
        "assoc_topk_mix": assoc_topk_mix,
        "assoc_read_gate": read_gate,
        "assoc_write_gate": write_gate,
        "assoc_lm_head": assoc_lm_head,
    }


def model_component_flops(
    *,
    model_name: str,
    seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    vocab_size: int,
) -> Dict[str, int]:
    if model_name == "transformer":
        return transformer_forward_flops(seq_len, d_model, n_layers, vocab_size)

    if not model_name.startswith("triple-"):
        raise ValueError(f"Unsupported model for this estimator: {model_name}")

    cfg = triple_latent_config(
        model_name,
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
    )

    components = triple_hybrid_forward_flops(
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
        vocab_size=vocab_size,
    )

    if cfg.use_assoc_memory:
        components.update(
            assoc_memory_flops(
                seq_len=seq_len,
                d_model=d_model,
                assoc_dim=cfg.assoc_dim,
                assoc_topk=cfg.assoc_topk,
                layer_multiplier=n_layers if cfg.assoc_layer_mode == "all" else 1,
                use_assoc_read_gate=cfg.use_assoc_read_gate,
                use_assoc_write_gate=cfg.use_assoc_write_gate,
                logits_fusion=cfg.assoc_fusion == "logits",
                vocab_size=vocab_size,
            )
        )

    return components


def estimate_compute(
    *,
    output_dir: str | Path,
    models: List[str],
    seq_lens: List[int],
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    vocab_size: int,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    totals_by_seq: Dict[int, Dict[str, int]] = {}

    for seq_len in seq_lens:
        totals_by_seq[seq_len] = {}
        for model_name in models:
            components = model_component_flops(
                model_name=model_name,
                seq_len=seq_len,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                state_dim=state_dim,
                vocab_size=vocab_size,
            )
            total_flops = sum(components.values())
            totals_by_seq[seq_len][model_name] = total_flops
            row: Dict[str, object] = {
                "model": model_name,
                "seq_len": seq_len,
                "total_flops": total_flops,
                "total_mflops": round(total_flops / 1e6, 3),
            }
            row.update(components)
            rows.append(row)

    for row in rows:
        baseline = totals_by_seq[int(row["seq_len"])]["transformer"]
        row["relative_to_transformer"] = round(float(row["total_flops"]) / float(baseline), 4)

    payload = {
        "assumptions": [
            "Counts use 2 FLOPs per multiply-add and track only dominant linear/attention/einsum terms.",
            "Embedding lookups, layer norm, GELU, softmax, top-k selection, and Python overhead are omitted.",
            "Triple-hybrid includes the front-end local convolution; logits-fusion variants include the extra vocab projection.",
        ],
        "config": {
            "models": models,
            "seq_lens": seq_lens,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "state_dim": state_dim,
            "vocab_size": vocab_size,
        },
        "rows": rows,
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "compute_estimate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(root / "compute_estimate.csv", rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate dominant forward FLOPs for the generic triple-latent paper.")
    parser.add_argument("--output-dir", default="auto_research_llm_ideas/results/_generic_triple_compute_d64_v1")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[64, 128, 512])
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=BYTE_VOCAB_SIZE)
    args = parser.parse_args()

    payload = estimate_compute(
        output_dir=args.output_dir,
        models=args.models,
        seq_lens=args.seq_lens,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        state_dim=args.state_dim,
        vocab_size=args.vocab_size,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
