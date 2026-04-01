from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from auto_research_llm_ideas.experiments.data import default_device
from auto_research_llm_ideas.experiments.run_full_suite import (
    compare_against_standard_attention,
    run_suite,
    write_csv,
)
from auto_research_llm_ideas.experiments.train_lm import train_language_model


MATCHED_WIDTH_MODELS = ["transformer", "triple-latent", "triple-slot", "triple-hybrid"]

# Approximate parameter matching to the d_model=64 Transformer baseline.
PARAM_MATCHED_LM_CONFIGS = [
    {"model": "transformer", "d_model": 64},
    {"model": "triple-latent", "d_model": 60},
    {"model": "triple-slot", "d_model": 60},
    {"model": "triple-hybrid", "d_model": 56},
]


def run_param_matched_lm_followup(
    *,
    output_root: str | Path,
    steps: int,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    dropout: float,
) -> Dict[str, object]:
    device = default_device()
    root = Path(output_root)

    results: List[Dict[str, object]] = []
    for config in PARAM_MATCHED_LM_CONFIGS:
        result = train_language_model(
            model_name=str(config["model"]),
            dataset_name="wikitext2",
            output_dir=root / "lm",
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            eval_batches=eval_batches,
            d_model=int(config["d_model"]),
            n_layers=n_layers,
            n_heads=n_heads,
            state_dim=state_dim,
            dropout=dropout,
            device=device,
        )
        result["d_model"] = int(config["d_model"])
        results.append(result)

    comparisons = compare_against_standard_attention(
        results,
        key_fields=("dataset",),
        lower_is_better=("val_loss", "val_bpb", "val_perplexity"),
        ratio_fields=("num_params", "train_seconds"),
    )

    write_csv(root / "lm_results.csv", results)
    write_csv(root / "lm_vs_standard_attention.csv", comparisons)

    summary = {
        "device": str(device),
        "configs": PARAM_MATCHED_LM_CONFIGS,
        "lm_results": results,
        "lm_vs_standard_attention": comparisons,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_followup(
    *,
    output_root: str | Path,
    lm_steps: int,
    memory_steps: int,
    bench_seq_lens: List[int],
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    dropout: float,
    lm_batch_size: int,
    memory_batch_size: int,
    seq_len: int,
    eval_batches: int,
) -> Dict[str, object]:
    root = Path(output_root)
    del memory_batch_size
    matched_width = run_suite(
        output_root=root / "matched_width",
        models=MATCHED_WIDTH_MODELS,
        lm_steps=lm_steps,
        memory_steps=memory_steps,
        bench_seq_lens=bench_seq_lens,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
    )
    param_matched = run_param_matched_lm_followup(
        output_root=root / "param_matched_lm",
        steps=lm_steps,
        batch_size=lm_batch_size,
        seq_len=seq_len,
        eval_batches=eval_batches,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
        dropout=dropout,
    )

    summary = {
        "matched_width": matched_width,
        "param_matched_lm": param_matched,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the generic triple-latent follow-up against a Transformer baseline."
    )
    parser.add_argument("--output-root", default="auto_research_llm_ideas/results/generic_latent_followup")
    parser.add_argument("--lm-steps", type=int, default=80)
    parser.add_argument("--memory-steps", type=int, default=200)
    parser.add_argument("--bench-seq-lens", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lm-batch-size", type=int, default=16)
    parser.add_argument("--memory-batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=128)
    args = parser.parse_args()

    summary = run_followup(
        output_root=args.output_root,
        lm_steps=args.lm_steps,
        memory_steps=args.memory_steps,
        bench_seq_lens=args.bench_seq_lens,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        state_dim=args.state_dim,
        dropout=args.dropout,
        lm_batch_size=args.lm_batch_size,
        memory_batch_size=args.memory_batch_size,
        seq_len=args.seq_len,
        eval_batches=args.eval_batches,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
