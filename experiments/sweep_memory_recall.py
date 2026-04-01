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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_summary(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, int, int, int, int], List[Dict[str, object]]] = {}
    for row in results:
        key = (
            str(row["model"]),
            int(row["search_d_model"]),
            int(row["search_n_layers"]),
            int(row["search_state_dim"]),
            int(row["steps"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, rows in grouped.items():
        model, d_model, n_layers, state_dim, steps = key
        summary_rows.append(
            {
                "model": model,
                "d_model": d_model,
                "n_layers": n_layers,
                "state_dim": state_dim,
                "steps": steps,
                "runs": len(rows),
                "mean_eval_accuracy": sum(float(row["eval_accuracy"]) for row in rows) / len(rows),
                "mean_eval_loss": sum(float(row["eval_loss"]) for row in rows) / len(rows),
                "mean_train_seconds": sum(float(row["train_seconds"]) for row in rows) / len(rows),
                "num_params": int(rows[0]["num_params"]),
            }
        )

    summary_rows.sort(key=lambda row: (-float(row["mean_eval_accuracy"]), float(row["mean_eval_loss"])))
    return summary_rows


def sweep_memory_recall(
    *,
    output_root: str | Path,
    models: List[str],
    d_models: List[int],
    n_layers_list: List[int],
    state_dims: List[int],
    steps_list: List[int],
    seeds: List[int],
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    n_heads: int,
    dropout: float,
    num_pairs: int,
    lr_warmup_steps: int,
    curriculum_start_seq_len: int | None,
    curriculum_steps: int,
    curriculum_start_num_pairs: int | None,
    read_gate_bias_init: float | None,
    write_gate_bias_init: float | None,
) -> Dict[str, object]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for model_name in models:
        for d_model in d_models:
            for n_layers in n_layers_list:
                for state_dim in state_dims:
                    for steps in steps_list:
                        for seed in seeds:
                            run_dir = (
                                root
                                / "memory"
                                / f"{model_name}_d{d_model}_l{n_layers}_s{state_dim}_steps{steps}_seed{seed}"
                            )
                            result = train_memory_model(
                                model_name=model_name,
                                output_dir=run_dir,
                                steps=steps,
                                batch_size=batch_size,
                                seq_len=seq_len,
                                eval_batches=eval_batches,
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
                            result["search_d_model"] = d_model
                            result["search_n_layers"] = n_layers
                            result["search_state_dim"] = state_dim
                            results.append(result)

    results.sort(key=lambda row: (-float(row["eval_accuracy"]), float(row["eval_loss"])))
    summary_rows = group_summary(results)

    write_csv(root / "memory_results.csv", results)
    write_csv(root / "summary.csv", summary_rows)

    summary = {
        "models": models,
        "d_models": d_models,
        "n_layers_list": n_layers_list,
        "state_dims": state_dims,
        "steps_list": steps_list,
        "seeds": seeds,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "eval_batches": eval_batches,
        "n_heads": n_heads,
        "dropout": dropout,
        "num_pairs": num_pairs,
        "lr_warmup_steps": lr_warmup_steps,
        "curriculum_start_seq_len": curriculum_start_seq_len,
        "curriculum_steps": curriculum_steps,
        "curriculum_start_num_pairs": curriculum_start_num_pairs,
        "read_gate_bias_init": read_gate_bias_init,
        "write_gate_bias_init": write_gate_bias_init,
        "results": results,
        "summary_rows": summary_rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep associative-recall experiments.")
    parser.add_argument("--output-root", default="auto_research_llm_ideas/results/memory_recall_sweep")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--d-models", nargs="+", type=int, required=True)
    parser.add_argument("--n-layers-list", nargs="+", type=int, default=[3])
    parser.add_argument("--state-dims", nargs="+", type=int, default=[8])
    parser.add_argument("--steps-list", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[19])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-pairs", type=int, default=4)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--curriculum-start-seq-len", type=int, default=None)
    parser.add_argument("--curriculum-steps", type=int, default=0)
    parser.add_argument("--curriculum-start-num-pairs", type=int, default=None)
    parser.add_argument("--read-gate-bias-init", type=float, default=None)
    parser.add_argument("--write-gate-bias-init", type=float, default=None)
    args = parser.parse_args()

    summary = sweep_memory_recall(
        output_root=args.output_root,
        models=args.models,
        d_models=args.d_models,
        n_layers_list=args.n_layers_list,
        state_dims=args.state_dims,
        steps_list=args.steps_list,
        seeds=args.seeds,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_batches=args.eval_batches,
        n_heads=args.n_heads,
        dropout=args.dropout,
        num_pairs=args.num_pairs,
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
