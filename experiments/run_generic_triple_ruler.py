from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from auto_research_llm_ideas.experiments.data import default_device
from auto_research_llm_ideas.experiments.ruler_subset import LocalCheckpointGenerator, evaluate_ruler_core


DEFAULT_CHECKPOINTS = {
    "transformer": "auto_research_llm_ideas/results/_gated_recall_d64_tradeoffs_v1/lm/transformer/transformer_lm.pt",
    "triple-hybrid": "auto_research_llm_ideas/results/_gated_recall_d64_tradeoffs_v1/lm/triple-hybrid/triple-hybrid_lm.pt",
    "triple-hybrid-assoc-gated-last-logits": (
        "auto_research_llm_ideas/results/_gated_recall_d64_tradeoffs_v1/lm/"
        "triple-hybrid-assoc-gated-last-logits/triple-hybrid-assoc-gated-last-logits_lm.pt"
    ),
}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_null_fraction(value: str) -> tuple[int, int]:
    left, right = value.split("/")
    return int(left), int(right)


def load_or_run_summary(
    *,
    model_name: str,
    checkpoint_path: str,
    output_dir: Path,
    tasks: List[str],
    num_samples: int,
    max_seq_length: int | None,
    seed: int,
    skip_existing: bool,
) -> Dict[str, object]:
    summary_path = output_dir / "ruler_core_summary.json"
    if skip_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    generator = LocalCheckpointGenerator(checkpoint_path, device=default_device())
    return evaluate_ruler_core(
        generator=generator,
        tasks=tasks,
        num_samples=num_samples,
        max_seq_length=max_seq_length,
        output_dir=output_dir,
        seed=seed,
    )


def run_generic_triple_ruler(
    *,
    output_dir: str | Path,
    checkpoints: Dict[str, str],
    tasks: List[str],
    num_samples: int,
    max_seq_length: int | None,
    seed: int,
    skip_existing: bool,
) -> Dict[str, object]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    combined_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for model_name, checkpoint_path in checkpoints.items():
        model_output_dir = root / model_name
        summary = load_or_run_summary(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            output_dir=model_output_dir,
            tasks=tasks,
            num_samples=num_samples,
            max_seq_length=max_seq_length,
            seed=seed,
            skip_existing=skip_existing,
        )

        null_count = 0
        total_predictions = 0
        for metric in summary["task_metrics"]:
            nulls, total = parse_null_fraction(str(metric["null_predictions"]))
            null_count += nulls
            total_predictions += total
            combined_rows.append(
                {
                    "model": model_name,
                    "task": str(metric["task"]),
                    "task_type": str(metric["task_type"]),
                    "score": float(metric["score"]),
                    "null_predictions": str(metric["null_predictions"]),
                    "num_samples": int(metric["num_samples"]),
                    "seq_budget": int(metric["seq_budget"]),
                }
            )

        summary_rows.append(
            {
                "model": model_name,
                "checkpoint_path": checkpoint_path,
                "average_score": float(summary["average_score"]),
                "null_predictions_total": f"{null_count}/{total_predictions}",
                "seq_budget": int(summary["seq_budget"]),
                "num_samples": int(summary["num_samples"]),
            }
        )

    payload = {
        "tasks": tasks,
        "num_samples": num_samples,
        "max_seq_length": max_seq_length,
        "seed": seed,
        "summary_rows": summary_rows,
        "task_rows": combined_rows,
    }
    (root / "combined_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(root / "combined_metrics.csv", combined_rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or aggregate the generic triple-latent RULER-core comparison.")
    parser.add_argument("--output-dir", default="auto_research_llm_ideas/results/_generic_triple_ruler_d64_v1")
    parser.add_argument("--tasks", nargs="+", default=["niah_single_1", "vt", "cwe", "fwe"])
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    summary = run_generic_triple_ruler(
        output_dir=args.output_dir,
        checkpoints=DEFAULT_CHECKPOINTS,
        tasks=args.tasks,
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
