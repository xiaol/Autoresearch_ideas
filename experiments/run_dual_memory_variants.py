from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from auto_research_llm_ideas.experiments.benchmark import benchmark_models
from auto_research_llm_ideas.experiments.data import default_device
from auto_research_llm_ideas.experiments.train_lm import build_model, train_language_model
from auto_research_llm_ideas.experiments.train_memory import train_memory_model


DEFAULT_MODELS = [
    "transformer",
    "triple-hybrid",
    "triple-hybrid-assoc-sum",
    "triple-hybrid-assoc-concat",
    "triple-hybrid-assoc-logits",
    "triple-hybrid-assoc-serial-sum",
    "triple-hybrid-assoc-last-sum",
    "triple-hybrid-assoc-last-logits",
    "triple-hybrid-assoc-prev-sum",
    "triple-hybrid-assoc-prev-logits",
]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def estimate_params(
    *,
    model_name: str,
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    seq_len: int,
) -> int:
    model = build_model(
        model_name=model_name,
        vocab_size=256,
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
        dropout=0.1,
        pad_token_id=256,
    )
    return int(model.get_num_params())


def run_variants(
    *,
    output_root: str | Path,
    models: List[str],
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    dropout: float,
    lm_steps: int,
    lm_batch_size: int,
    memory_steps: int,
    memory_batch_size: int,
    seq_len: int,
    eval_batches: int,
    bench_seq_lens: List[int],
) -> Dict[str, object]:
    device = default_device()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    param_rows = []
    lm_results = []
    memory_results = []
    for model_name in models:
        param_rows.append(
            {
                "model": model_name,
                "estimated_params": estimate_params(
                    model_name=model_name,
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    state_dim=state_dim,
                    seq_len=seq_len,
                ),
            }
        )
        lm_result = train_language_model(
            model_name=model_name,
            dataset_name="wikitext2",
            output_dir=root / "lm" / model_name,
            steps=lm_steps,
            batch_size=lm_batch_size,
            seq_len=seq_len,
            eval_batches=eval_batches,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            state_dim=state_dim,
            dropout=dropout,
            device=device,
        )
        lm_results.append(lm_result)

        memory_result = train_memory_model(
            model_name=model_name,
            output_dir=root / "memory" / model_name,
            steps=memory_steps,
            batch_size=memory_batch_size,
            seq_len=seq_len,
            eval_batches=eval_batches,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            state_dim=state_dim,
            dropout=dropout,
            device=device,
        )
        memory_results.append(memory_result)

    benchmark_results = benchmark_models(
        models=models,
        output_dir=root / "benchmark",
        seq_lens=bench_seq_lens,
        batch_size=8,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
        dropout=dropout,
        device=device,
    )

    bench_by_model_seq = {
        (str(row["model"]), int(row["seq_len"])): row for row in benchmark_results
    }
    memory_by_model = {str(row["model"]): row for row in memory_results}
    lm_by_model = {str(row["model"]): row for row in lm_results}

    summary_rows = []
    for params_row in param_rows:
        model_name = str(params_row["model"])
        lm_row = lm_by_model[model_name]
        memory_row = memory_by_model[model_name]
        row = {
            "model": model_name,
            "num_params": int(params_row["estimated_params"]),
            "lm_val_bpb": float(lm_row["val_bpb"]),
            "lm_train_seconds": float(lm_row["train_seconds"]),
            "memory_accuracy": float(memory_row["eval_accuracy"]),
            "memory_train_seconds": float(memory_row["train_seconds"]),
        }
        for seq_len_value in bench_seq_lens:
            bench_row = bench_by_model_seq[(model_name, seq_len_value)]
            row[f"tok_per_s_seq{seq_len_value}"] = float(bench_row["tokens_per_second"])
        summary_rows.append(row)

    summary_rows.sort(key=lambda row: (-float(row["memory_accuracy"]), float(row["lm_val_bpb"])))

    write_csv(root / "params.csv", param_rows)
    write_csv(root / "lm_results.csv", lm_results)
    write_csv(root / "memory_results.csv", memory_results)
    write_csv(root / "summary.csv", summary_rows)

    summary = {
        "device": str(device),
        "models": models,
        "config": {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "state_dim": state_dim,
            "dropout": dropout,
            "lm_steps": lm_steps,
            "lm_batch_size": lm_batch_size,
            "memory_steps": memory_steps,
            "memory_batch_size": memory_batch_size,
            "seq_len": seq_len,
            "eval_batches": eval_batches,
            "bench_seq_lens": bench_seq_lens,
        },
        "summary_rows": summary_rows,
        "lm_results": lm_results,
        "memory_results": memory_results,
        "benchmark_results": benchmark_results,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dual-memory triple variants.")
    parser.add_argument("--output-root", default="auto_research_llm_ideas/results/dual_memory_variants")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lm-steps", type=int, default=80)
    parser.add_argument("--lm-batch-size", type=int, default=16)
    parser.add_argument("--memory-steps", type=int, default=200)
    parser.add_argument("--memory-batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--bench-seq-lens", nargs="+", type=int, default=[64, 512])
    args = parser.parse_args()

    summary = run_variants(
        output_root=args.output_root,
        models=args.models,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        state_dim=args.state_dim,
        dropout=args.dropout,
        lm_steps=args.lm_steps,
        lm_batch_size=args.lm_batch_size,
        memory_steps=args.memory_steps,
        memory_batch_size=args.memory_batch_size,
        seq_len=args.seq_len,
        eval_batches=args.eval_batches,
        bench_seq_lens=args.bench_seq_lens,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
