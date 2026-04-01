from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from auto_research_llm_ideas.experiments.data import BYTE_PAD, BYTE_VOCAB_SIZE
from auto_research_llm_ideas.experiments.train_lm import build_model, train_language_model


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
        vocab_size=BYTE_VOCAB_SIZE,
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
        dropout=0.1,
        pad_token_id=BYTE_PAD,
    )
    return int(model.get_num_params())


def sweep(
    *,
    output_root: str | Path,
    models: List[str],
    d_models: List[int],
    state_dims: List[int],
    n_layers_list: List[int],
    n_heads: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    max_params: int | None,
    baseline_bpb: float | None,
) -> Dict[str, object]:
    root = Path(output_root)
    candidates: List[Dict[str, object]] = []

    for model_name in models:
        for d_model in d_models:
            if d_model % n_heads != 0:
                continue
            for state_dim in state_dims:
                for n_layers in n_layers_list:
                    num_params = estimate_params(
                        model_name=model_name,
                        d_model=d_model,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        state_dim=state_dim,
                        seq_len=seq_len,
                    )
                    if max_params is not None and num_params > max_params:
                        continue
                    candidates.append(
                        {
                            "model": model_name,
                            "d_model": d_model,
                            "state_dim": state_dim,
                            "n_layers": n_layers,
                            "estimated_params": num_params,
                        }
                    )

    write_csv(root / "candidate_grid.csv", candidates)

    results: List[Dict[str, object]] = []
    for candidate in candidates:
        model_name = str(candidate["model"])
        d_model = int(candidate["d_model"])
        state_dim = int(candidate["state_dim"])
        n_layers = int(candidate["n_layers"])
        run_dir = root / "lm" / f"{model_name}_d{d_model}_s{state_dim}_l{n_layers}"
        result = train_language_model(
            model_name=model_name,
            dataset_name="wikitext2",
            output_dir=run_dir,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            eval_batches=eval_batches,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            state_dim=state_dim,
        )
        result["search_d_model"] = d_model
        result["search_state_dim"] = state_dim
        result["search_n_layers"] = n_layers
        if baseline_bpb is not None:
            result["baseline_val_bpb"] = baseline_bpb
            result["val_bpb_improvement_vs_baseline"] = baseline_bpb - float(result["val_bpb"])
        results.append(result)

    results.sort(key=lambda row: float(row["val_bpb"]))
    write_csv(root / "sweep_results.csv", results)

    summary = {
        "models": models,
        "d_models": d_models,
        "state_dims": state_dims,
        "n_layers_list": n_layers_list,
        "n_heads": n_heads,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "eval_batches": eval_batches,
        "max_params": max_params,
        "baseline_bpb": baseline_bpb,
        "num_candidates": len(candidates),
        "results": results,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep triple-latent variants on byte-level WikiText-2.")
    parser.add_argument("--output-root", default="auto_research_llm_ideas/results/triple_latent_sweep")
    parser.add_argument("--models", nargs="+", default=["triple-latent", "triple-slot", "triple-hybrid", "triple-slot-hybrid"])
    parser.add_argument("--d-models", nargs="+", type=int, default=[56, 60, 64])
    parser.add_argument("--state-dims", nargs="+", type=int, default=[8, 12, 16])
    parser.add_argument("--n-layers-list", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--max-params", type=int, default=None)
    parser.add_argument("--baseline-bpb", type=float, default=None)
    args = parser.parse_args()

    summary = sweep(
        output_root=args.output_root,
        models=args.models,
        d_models=args.d_models,
        state_dims=args.state_dims,
        n_layers_list=args.n_layers_list,
        n_heads=args.n_heads,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_batches=args.eval_batches,
        max_params=args.max_params,
        baseline_bpb=args.baseline_bpb,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
