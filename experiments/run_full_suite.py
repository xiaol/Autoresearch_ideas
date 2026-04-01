from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt

from auto_research_llm_ideas.experiments.benchmark import benchmark_models
from auto_research_llm_ideas.experiments.data import default_device
from auto_research_llm_ideas.experiments.ruler_subset import evaluate_ruler_core, LocalCheckpointGenerator
from auto_research_llm_ideas.experiments.train_lm import train_language_model
from auto_research_llm_ideas.experiments.train_memory import train_memory_model


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_lm_results(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [row["model"] for row in rows]
    values = [row["val_bpb"] for row in rows]

    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(names, values, color=["#7f8c8d", "#d35400", "#16a085", "#2c3e50"])
    plt.ylabel("Validation bits-per-byte")
    plt.title("Byte-Level LM Pilot on WikiText-2")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_memory_results(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [row["model"] for row in rows]
    values = [row["eval_accuracy"] for row in rows]

    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(names, values, color=["#7f8c8d", "#d35400", "#16a085", "#2c3e50"])
    plt.ylabel("Associative recall accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Long-Range Memory Stress Test")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_benchmark(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        grouped.setdefault(row["model"], {"seq": [], "tps": []})
        grouped[row["model"]]["seq"].append(row["seq_len"])
        grouped[row["model"]]["tps"].append(row["tokens_per_second"])

    plt.figure(figsize=(8, 4.8))
    for model_name, series in grouped.items():
        paired = sorted(zip(series["seq"], series["tps"]))
        plt.plot([item[0] for item in paired], [item[1] for item in paired], marker="o", linewidth=2, label=model_name)
    plt.xlabel("Sequence length")
    plt.ylabel("Tokens / second")
    plt.title("Throughput Scaling Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _baseline_key(row: Dict[str, object], fields: Sequence[str]) -> Tuple[object, ...]:
    return tuple(row[field] for field in fields)


def compare_against_standard_attention(
    rows: List[Dict[str, object]],
    *,
    key_fields: Sequence[str],
    lower_is_better: Sequence[str] = (),
    higher_is_better: Sequence[str] = (),
    ratio_fields: Sequence[str] = (),
    baseline_model: str = "transformer",
) -> List[Dict[str, object]]:
    baseline_rows = {
        _baseline_key(row, key_fields): row
        for row in rows
        if row.get("model") == baseline_model
    }
    comparisons: List[Dict[str, object]] = []

    for row in rows:
        if row.get("model") == baseline_model:
            continue
        baseline = baseline_rows.get(_baseline_key(row, key_fields))
        if baseline is None:
            continue

        result: Dict[str, object] = {
            field: row[field]
            for field in key_fields
        }
        result["model"] = row["model"]
        result["baseline_model"] = baseline_model

        for metric in lower_is_better:
            value = float(row[metric])
            baseline_value = float(baseline[metric])
            result[metric] = value
            result[f"{metric}_{baseline_model}"] = baseline_value
            result[f"{metric}_delta"] = value - baseline_value
            result[f"{metric}_improvement"] = baseline_value - value

        for metric in higher_is_better:
            value = float(row[metric])
            baseline_value = float(baseline[metric])
            result[metric] = value
            result[f"{metric}_{baseline_model}"] = baseline_value
            result[f"{metric}_delta"] = value - baseline_value
            result[f"{metric}_improvement"] = value - baseline_value

        for metric in ratio_fields:
            value = float(row[metric])
            baseline_value = float(baseline[metric])
            result[metric] = value
            result[f"{metric}_{baseline_model}"] = baseline_value
            result[f"{metric}_ratio"] = value / baseline_value if baseline_value else None

        comparisons.append(result)

    return comparisons


def run_suite(
    output_root: str | Path,
    models: List[str],
    lm_steps: int,
    memory_steps: int,
    bench_seq_lens: List[int],
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    ruler_tasks: List[str] | None = None,
    ruler_samples: int = 0,
    ruler_max_seq_length: int | None = None,
) -> Dict[str, object]:
    device = default_device()
    root = Path(output_root)
    figures_dir = root / "figures"

    lm_results = []
    for model_name in models:
        lm_results.append(
            train_language_model(
                model_name=model_name,
                dataset_name="wikitext2",
                output_dir=root / "lm",
                steps=lm_steps,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                state_dim=state_dim,
                device=device,
            )
        )

    memory_results = []
    for model_name in models:
        memory_results.append(
            train_memory_model(
                model_name=model_name,
                output_dir=root / "memory",
                steps=memory_steps,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                state_dim=state_dim,
                device=device,
            )
        )

    benchmark_results = benchmark_models(
        models=models,
        output_dir=root / "benchmark",
        seq_lens=bench_seq_lens,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
        device=device,
    )

    ruler_results = []
    if ruler_tasks:
        for result in lm_results:
            checkpoint_path = result.get("checkpoint_path")
            if checkpoint_path is None:
                continue
            generator = LocalCheckpointGenerator(checkpoint_path, device=device)
            ruler_summary = evaluate_ruler_core(
                generator=generator,
                tasks=ruler_tasks,
                num_samples=ruler_samples,
                max_seq_length=ruler_max_seq_length,
                output_dir=root / "ruler" / result["model"],
                seed=17,
            )
            ruler_results.append(
                {
                    "model": result["model"],
                    "seq_budget": ruler_summary["seq_budget"],
                    "num_samples": ruler_summary["num_samples"],
                    "average_score": ruler_summary["average_score"],
                    "tasks": ",".join(ruler_summary["tasks"]),
                }
            )

    write_csv(root / "lm_results.csv", lm_results)
    write_csv(root / "memory_results.csv", memory_results)
    write_csv(root / "benchmark_results.csv", benchmark_results)
    if ruler_results:
        write_csv(root / "ruler_results.csv", ruler_results)

    lm_vs_baseline = compare_against_standard_attention(
        lm_results,
        key_fields=("dataset",),
        lower_is_better=("val_loss", "val_bpb", "val_perplexity"),
        ratio_fields=("num_params", "train_seconds"),
    )
    memory_vs_baseline = compare_against_standard_attention(
        memory_results,
        key_fields=("task", "seq_len"),
        lower_is_better=("eval_loss",),
        higher_is_better=("eval_accuracy",),
        ratio_fields=("num_params", "train_seconds"),
    )
    benchmark_vs_baseline = compare_against_standard_attention(
        benchmark_results,
        key_fields=("task", "seq_len", "batch_size"),
        lower_is_better=("milliseconds_per_iter", "memory_mb"),
        higher_is_better=("tokens_per_second",),
        ratio_fields=("num_params",),
    )
    ruler_vs_baseline = compare_against_standard_attention(
        ruler_results,
        key_fields=("tasks", "seq_budget", "num_samples"),
        higher_is_better=("average_score",),
    )
    write_csv(root / "lm_vs_standard_attention.csv", lm_vs_baseline)
    write_csv(root / "memory_vs_standard_attention.csv", memory_vs_baseline)
    write_csv(root / "benchmark_vs_standard_attention.csv", benchmark_vs_baseline)
    if ruler_vs_baseline:
        write_csv(root / "ruler_vs_standard_attention.csv", ruler_vs_baseline)

    plot_lm_results(figures_dir / "lm_validation_bpb.png", lm_results)
    plot_memory_results(figures_dir / "memory_accuracy.png", memory_results)
    plot_benchmark(figures_dir / "throughput_scaling.png", benchmark_results)

    summary = {
        "device": str(device),
        "models": models,
        "lm_results": lm_results,
        "memory_results": memory_results,
        "benchmark_results": benchmark_results,
        "ruler_results": ruler_results,
        "lm_vs_standard_attention": lm_vs_baseline,
        "memory_vs_standard_attention": memory_vs_baseline,
        "benchmark_vs_standard_attention": benchmark_vs_baseline,
        "ruler_vs_standard_attention": ruler_vs_baseline,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full UniMatrix pilot experiment suite.")
    parser.add_argument("--output-root", default="auto_research_llm_ideas/results")
    parser.add_argument("--models", nargs="+", default=["transformer", "unimatrix-core", "unimatrix-rosa", "unimatrix-discovery"])
    parser.add_argument("--lm-steps", type=int, default=150)
    parser.add_argument("--memory-steps", type=int, default=250)
    parser.add_argument("--bench-seq-lens", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--ruler-tasks", nargs="+", default=[])
    parser.add_argument("--ruler-samples", type=int, default=8)
    parser.add_argument("--ruler-max-seq-length", type=int, default=None)
    args = parser.parse_args()

    summary = run_suite(
        output_root=args.output_root,
        models=args.models,
        lm_steps=args.lm_steps,
        memory_steps=args.memory_steps,
        bench_seq_lens=args.bench_seq_lens,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        state_dim=args.state_dim,
        ruler_tasks=args.ruler_tasks,
        ruler_samples=args.ruler_samples,
        ruler_max_seq_length=args.ruler_max_seq_length,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
