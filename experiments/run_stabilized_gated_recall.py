from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from auto_research_llm_ideas.experiments.train_memory import train_memory_model


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


def summarize_results(rows: List[Dict[str, object]]) -> Dict[str, object]:
    mean_accuracy = sum(float(row["eval_accuracy"]) for row in rows) / max(len(rows), 1)
    mean_loss = sum(float(row["eval_loss"]) for row in rows) / max(len(rows), 1)
    best_accuracy = max(float(row["eval_accuracy"]) for row in rows)
    return {
        "runs": len(rows),
        "mean_eval_accuracy": mean_accuracy,
        "mean_eval_loss": mean_loss,
        "best_eval_accuracy": best_accuracy,
    }


def run_stabilized_gated_recall(
    *,
    output_root: str | Path,
    model_name: str,
    seeds: List[int],
    steps: int,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    dropout: float,
    num_pairs: int,
    lr: float,
    lr_warmup_steps: int,
    curriculum_start_seq_len: int | None,
    curriculum_steps: int,
    curriculum_start_num_pairs: int | None,
    read_gate_bias_init: float | None,
    write_gate_bias_init: float | None,
) -> Dict[str, object]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        run_dir = root / f"seed_{seed}"
        result = train_memory_model(
            model_name=model_name,
            output_dir=run_dir,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            eval_batches=eval_batches,
            lr=lr,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            state_dim=state_dim,
            dropout=dropout,
            seed=seed,
            num_pairs=num_pairs,
            lr_warmup_steps=lr_warmup_steps,
            curriculum_start_seq_len=curriculum_start_seq_len,
            curriculum_steps=curriculum_steps,
            curriculum_start_num_pairs=curriculum_start_num_pairs,
            read_gate_bias_init=read_gate_bias_init,
            write_gate_bias_init=write_gate_bias_init,
        )
        rows.append(result)

    summary = {
        "config": {
            "model_name": model_name,
            "seeds": seeds,
            "steps": steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "eval_batches": eval_batches,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "state_dim": state_dim,
            "dropout": dropout,
            "num_pairs": num_pairs,
            "lr": lr,
            "lr_warmup_steps": lr_warmup_steps,
            "curriculum_start_seq_len": curriculum_start_seq_len,
            "curriculum_steps": curriculum_steps,
            "curriculum_start_num_pairs": curriculum_start_num_pairs,
            "read_gate_bias_init": read_gate_bias_init,
            "write_gate_bias_init": write_gate_bias_init,
        },
        "results": rows,
        "summary": summarize_results(rows),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(root / "results.csv", rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stabilized associative-recall follow-ups for the gated triple-hybrid.")
    parser.add_argument("--output-root", default="auto_research_llm_ideas/results/stabilized_gated_recall")
    parser.add_argument("--model-name", default="triple-hybrid-assoc-gated-last-logits")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 29])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-pairs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=50)
    parser.add_argument("--curriculum-start-seq-len", type=int, default=32)
    parser.add_argument("--curriculum-steps", type=int, default=150)
    parser.add_argument("--curriculum-start-num-pairs", type=int, default=None)
    parser.add_argument("--read-gate-bias-init", type=float, default=-2.0)
    parser.add_argument("--write-gate-bias-init", type=float, default=None)
    args = parser.parse_args()

    summary = run_stabilized_gated_recall(
        output_root=args.output_root,
        model_name=args.model_name,
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_batches=args.eval_batches,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        state_dim=args.state_dim,
        dropout=args.dropout,
        num_pairs=args.num_pairs,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        curriculum_start_seq_len=args.curriculum_start_seq_len,
        curriculum_steps=args.curriculum_steps,
        curriculum_start_num_pairs=args.curriculum_start_num_pairs,
        read_gate_bias_init=args.read_gate_bias_init,
        write_gate_bias_init=args.write_gate_bias_init,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
