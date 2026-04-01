from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from auto_research_llm_ideas.experiments.data import default_device, set_seed
from auto_research_llm_ideas.experiments.train_memory import AR_PAD, AR_VOCAB_SIZE, sample_memory_batch
from auto_research_llm_ideas.model.unimatrix import UniMatrixLM, variant_config


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_memory_task_custom(
    model: torch.nn.Module,
    *,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    batches: int,
    seed: int,
    num_pairs: int,
) -> Dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    total_correct = 0
    total_examples = 0
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(batches):
            x, y = sample_memory_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                generator=generator,
                num_pairs=num_pairs,
            )
            logits, _ = model(x)
            final_logits = logits[:, -1, :]
            final_targets = y[:, -1]
            loss = F.cross_entropy(final_logits, final_targets)
            total_loss += loss.item() * batch_size
            total_correct += (final_logits.argmax(dim=-1) == final_targets).sum().item()
            total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def train_sparsepointer_variant(
    *,
    tag: str,
    output_dir: str | Path,
    steps: int,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    lr: float,
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    dropout: float,
    seed: int,
    train_num_pairs: int,
    eval_num_pairs_list: List[int],
    assoc_slots: int,
    use_pointer_logits: bool,
    use_pointer_write_gate: bool,
    device: torch.device | None = None,
) -> Dict[str, object]:
    device = device or default_device()
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    cfg = variant_config(
        "unimatrix-sparsepointer",
        vocab_size=AR_VOCAB_SIZE,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
        dropout=dropout,
        max_seq_len=seq_len,
        pad_token_id=AR_PAD,
        assoc_slots=assoc_slots,
        use_pointer_logits=use_pointer_logits,
        use_pointer_write_gate=use_pointer_write_gate,
        variant_name=tag,
    )
    model = UniMatrixLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    wall_start = time.perf_counter()
    last_loss = 0.0

    for step in range(1, steps + 1):
        model.train()
        x, y = sample_memory_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            generator=generator,
            num_pairs=train_num_pairs,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        final_logits = logits[:, -1, :]
        final_targets = y[:, -1]
        loss = F.cross_entropy(final_logits, final_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.item())

        if step % max(steps // 5, 1) == 0:
            print(f"[SP-ABL] {tag} step {step:04d}/{steps} loss={loss.item():.4f}")

    duration = time.perf_counter() - wall_start
    metrics_by_pairs: Dict[int, Dict[str, float]] = {}
    for eval_pairs in eval_num_pairs_list:
        metrics_by_pairs[eval_pairs] = evaluate_memory_task_custom(
            model,
            device=device,
            seq_len=seq_len,
            batch_size=batch_size,
            batches=eval_batches,
            seed=seed + 1 + eval_pairs,
            num_pairs=eval_pairs,
        )

    result: Dict[str, object] = {
        "model": "unimatrix-sparsepointer",
        "tag": tag,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "seed": seed,
        "device": str(device),
        "train_seconds": duration,
        "train_loss_last": last_loss,
        "num_params": int(model.get_num_params()),
        "train_num_pairs": train_num_pairs,
        "assoc_slots": assoc_slots,
        "use_pointer_logits": use_pointer_logits,
        "use_pointer_write_gate": use_pointer_write_gate,
        "dropout": dropout,
        "lr": lr,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "state_dim": state_dim,
    }
    for eval_pairs, metrics in metrics_by_pairs.items():
        result[f"eval_pairs_{eval_pairs}_loss"] = metrics["loss"]
        result[f"eval_pairs_{eval_pairs}_accuracy"] = metrics["accuracy"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def default_configs() -> List[Dict[str, object]]:
    return [
        {"tag": "slots4_ptr", "assoc_slots": 4, "use_pointer_logits": True, "use_pointer_write_gate": True},
        {"tag": "slots8_ptr", "assoc_slots": 8, "use_pointer_logits": True, "use_pointer_write_gate": True},
        {"tag": "slots16_ptr", "assoc_slots": 16, "use_pointer_logits": True, "use_pointer_write_gate": True},
        {"tag": "slots32_ptr", "assoc_slots": 32, "use_pointer_logits": True, "use_pointer_write_gate": True},
        {"tag": "slots32_noptr", "assoc_slots": 32, "use_pointer_logits": False, "use_pointer_write_gate": True},
        {"tag": "slots32_nogate", "assoc_slots": 32, "use_pointer_logits": True, "use_pointer_write_gate": False},
    ]


def summarize_results(results: List[Dict[str, object]], eval_num_pairs_list: List[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in results:
        summary_row: Dict[str, object] = {
            "tag": row["tag"],
            "assoc_slots": row["assoc_slots"],
            "use_pointer_logits": row["use_pointer_logits"],
            "use_pointer_write_gate": row["use_pointer_write_gate"],
            "num_params": row["num_params"],
            "train_seconds": round(float(row["train_seconds"]), 3),
        }
        for eval_pairs in eval_num_pairs_list:
            summary_row[f"acc_pairs_{eval_pairs}"] = round(float(row[f"eval_pairs_{eval_pairs}_accuracy"]), 6)
            summary_row[f"loss_pairs_{eval_pairs}"] = round(float(row[f"eval_pairs_{eval_pairs}_loss"]), 6)
        rows.append(summary_row)

    rows.sort(
        key=lambda row: (
            -float(row[f"acc_pairs_{eval_num_pairs_list[0]}"]),
            -float(row.get(f"acc_pairs_{eval_num_pairs_list[-1]}", 0.0)),
        )
    )
    return rows


def run_ablation_suite(
    *,
    output_root: str | Path,
    steps: int,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    lr: float,
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    dropout: float,
    seed: int,
    train_num_pairs: int,
    eval_num_pairs_list: List[int],
) -> Dict[str, object]:
    device = default_device()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for cfg in default_configs():
        result = train_sparsepointer_variant(
            output_dir=root / "runs",
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
            train_num_pairs=train_num_pairs,
            eval_num_pairs_list=eval_num_pairs_list,
            device=device,
            **cfg,
        )
        results.append(result)

    summary_rows = summarize_results(results, eval_num_pairs_list)
    write_csv(root / "summary.csv", summary_rows)
    summary = {
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "eval_batches": eval_batches,
        "lr": lr,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "state_dim": state_dim,
        "dropout": dropout,
        "seed": seed,
        "train_num_pairs": train_num_pairs,
        "eval_num_pairs_list": eval_num_pairs_list,
        "results": results,
        "summary_rows": summary_rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SparsePointer associative-recall ablations.")
    parser.add_argument("--output-root", default="auto_research_llm_ideas/results/sparsepointer_ablations")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--train-num-pairs", type=int, default=4)
    parser.add_argument("--eval-num-pairs", nargs="+", type=int, default=[4, 6, 8])
    args = parser.parse_args()
    if args.train_num_pairs > 8 or any(num_pairs > 8 for num_pairs in args.eval_num_pairs):
        raise ValueError("This benchmark currently supports at most 8 key-value pairs.")

    summary = run_ablation_suite(
        output_root=args.output_root,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_batches=args.eval_batches,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        state_dim=args.state_dim,
        dropout=args.dropout,
        seed=args.seed,
        train_num_pairs=args.train_num_pairs,
        eval_num_pairs_list=args.eval_num_pairs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
