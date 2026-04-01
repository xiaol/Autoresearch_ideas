from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from auto_research_llm_ideas.experiments.data import default_device
from auto_research_llm_ideas.experiments.ruler_subset import LocalCheckpointGenerator, evaluate_ruler_core
from auto_research_llm_ideas.experiments.train_lm import train_language_model


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


def run_long_context_transfer(
    *,
    output_root: str | Path,
    models: List[str],
    dataset_name: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    lr: float,
    lr_warmup_steps: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    dropout: float,
    seed: int,
    ruler_tasks: List[str],
    ruler_num_samples: int,
) -> Dict[str, object]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    device = default_device()

    lm_rows: List[Dict[str, object]] = []
    ruler_rows: List[Dict[str, object]] = []

    for model_name in models:
        lm_dir = root / "lm" / model_name
        lm_result = train_language_model(
            model_name=model_name,
            dataset_name=dataset_name,
            output_dir=lm_dir,
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
            lr_warmup_steps=lr_warmup_steps,
            device=device,
        )
        lm_rows.append(lm_result)

        checkpoint_path = str(lm_result["checkpoint_path"])
        generator = LocalCheckpointGenerator(checkpoint_path, device=device)
        ruler_summary = evaluate_ruler_core(
            generator=generator,
            tasks=ruler_tasks,
            num_samples=ruler_num_samples,
            max_seq_length=seq_len,
            output_dir=root / "ruler" / model_name,
            seed=17,
        )
        ruler_rows.append(
            {
                "model": model_name,
                "checkpoint_path": checkpoint_path,
                "average_score": float(ruler_summary["average_score"]),
                "seq_budget": int(ruler_summary["seq_budget"]),
                "num_samples": int(ruler_summary["num_samples"]),
            }
        )
        for metric in ruler_summary["task_metrics"]:
            ruler_rows.append(
                {
                    "model": model_name,
                    "task": str(metric["task"]),
                    "task_type": str(metric["task_type"]),
                    "score": float(metric["score"]),
                    "null_predictions": str(metric["null_predictions"]),
                    "seq_budget": int(metric["seq_budget"]),
                    "num_samples": int(metric["num_samples"]),
                }
            )

    summary = {
        "config": {
            "models": models,
            "dataset_name": dataset_name,
            "steps": steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "eval_batches": eval_batches,
            "lr": lr,
            "lr_warmup_steps": lr_warmup_steps,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "state_dim": state_dim,
            "dropout": dropout,
            "seed": seed,
            "ruler_tasks": ruler_tasks,
            "ruler_num_samples": ruler_num_samples,
        },
        "lm_results": lm_rows,
        "ruler_results": ruler_rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(root / "lm_results.csv", lm_rows)
    write_csv(root / "ruler_results.csv", ruler_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train longer-context LM checkpoints and evaluate zero-shot RULER-core transfer.")
    parser.add_argument("--output-root", default="auto_research_llm_ideas/results/long_context_transfer")
    parser.add_argument("--models", nargs="+", default=["transformer", "triple-hybrid-assoc-gated-last-logits"])
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "tinyshakespeare"])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ruler-tasks", nargs="+", default=["niah_single_1", "vt", "cwe", "fwe"])
    parser.add_argument("--ruler-num-samples", type=int, default=8)
    args = parser.parse_args()

    summary = run_long_context_transfer(
        output_root=args.output_root,
        models=args.models,
        dataset_name=args.dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_batches=args.eval_batches,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        state_dim=args.state_dim,
        dropout=args.dropout,
        seed=args.seed,
        ruler_tasks=args.ruler_tasks,
        ruler_num_samples=args.ruler_num_samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
