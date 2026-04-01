from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from auto_research_llm_ideas.experiments.data import default_device, set_seed
from auto_research_llm_ideas.experiments.run_higher_order_suite import (
    STANDARD_ATTENTION_MODEL,
    TRAIN_TASKS,
    TRIPLE_NUM_CLASSES,
    TRIPLE_NUM_TAGS,
    TRIPLE_NUM_TASKS,
    TRIPLE_NUM_VALUES,
    TRIPLE_QUERY,
    TRIPLE_ROLE_A,
    TRIPLE_ROLE_B,
    TRIPLE_ROLE_C,
    TRIPLE_VOCAB_SIZE,
    TRIPLE_A_START,
    TRIPLE_B_START,
    TRIPLE_C_START,
    TRIPLE_TAG_START,
    TRIPLE_TASK_START,
    benchmark_higher_order_models,
    build_lookup_table,
    build_standard_attention_comparison,
    evaluate_triple_task,
    sample_triple_batch,
    write_csv,
)
from auto_research_llm_ideas.model.higher_order import HIGHER_ORDER_MODELS, HigherOrderConfig, build_higher_order_model


def _weight_token(value: float) -> str:
    return str(value).replace(".", "p")


def _stage_checkpoint_name(stage_index: int, task_name: str) -> str:
    return f"stage_{stage_index + 1:02d}_{task_name}.pt"


def _stage_result_name(stage_index: int, task_name: str) -> str:
    return f"stage_{stage_index + 1:02d}_{task_name}.json"


def _normalize_stage_steps(stage_tasks: List[str], stage_steps: List[int]) -> List[int]:
    if len(stage_steps) == 1:
        return stage_steps * len(stage_tasks)
    if len(stage_steps) != len(stage_tasks):
        raise ValueError("stage-steps must provide either one value or exactly one value per stage task.")
    return stage_steps


def build_aux_heads(cfg: HigherOrderConfig) -> torch.nn.ModuleDict:
    return torch.nn.ModuleDict(
        {
            "a": torch.nn.Linear(cfg.d_model, TRIPLE_NUM_VALUES),
            "b": torch.nn.Linear(cfg.d_model, TRIPLE_NUM_VALUES),
            "c": torch.nn.Linear(cfg.d_model, TRIPLE_NUM_VALUES),
        }
    )


def save_stage_checkpoint(
    path: Path,
    *,
    model_name: str,
    cfg: HigherOrderConfig,
    model: torch.nn.Module,
    aux_heads: torch.nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
    metadata: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "cfg": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "aux_heads_state_dict": aux_heads.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_stage_checkpoint(
    path: Path,
    *,
    device: torch.device,
    lr: float,
) -> tuple[str, HigherOrderConfig, torch.nn.Module, torch.nn.ModuleDict, torch.optim.Optimizer, Dict[str, object]]:
    payload = torch.load(path, map_location=device)
    model_name = str(payload["model_name"])
    cfg = HigherOrderConfig(**payload["cfg"])
    model = build_higher_order_model(model_name, cfg).to(device)
    aux_heads = build_aux_heads(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    aux_heads.load_state_dict(payload["aux_heads_state_dict"])
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(aux_heads.parameters()), lr=lr, weight_decay=0.01)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return model_name, cfg, model, aux_heads, optimizer, dict(payload.get("metadata", {}))


def load_existing_stage_rows(stage_dir: Path, stage_tasks: List[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for stage_index, task_name in enumerate(stage_tasks):
        path = stage_dir / _stage_result_name(stage_index, task_name)
        if not path.exists():
            break
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def evaluate_all_tasks(
    *,
    model: torch.nn.Module,
    aux_heads: torch.nn.ModuleDict,
    eval_tasks: List[str],
    device: torch.device,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
    seed: int,
    lookup_table: torch.Tensor,
    num_tags: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for task_name in eval_tasks:
        task_metrics = evaluate_triple_task(
            model,
            task_name,
            device,
            seq_len,
            batch_size,
            eval_batches,
            seed,
            lookup_table,
            num_tags=num_tags,
            aux_heads=aux_heads,
        )
        safe_name = task_name.replace("-", "_")
        metrics[f"{safe_name}_loss"] = task_metrics["loss"]
        metrics[f"{safe_name}_accuracy"] = task_metrics["accuracy"]
        metrics[f"{safe_name}_aux_a_accuracy"] = task_metrics["aux_a_accuracy"]
        metrics[f"{safe_name}_aux_b_accuracy"] = task_metrics["aux_b_accuracy"]
        metrics[f"{safe_name}_aux_c_accuracy"] = task_metrics["aux_c_accuracy"]
    return metrics


def train_stage(
    *,
    model_name: str,
    model: torch.nn.Module,
    aux_heads: torch.nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
    task_name: str,
    stage_index: int,
    stage_steps: int,
    cumulative_steps: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    lookup_table: torch.Tensor,
    num_tags: int,
    aux_loss_weight: float,
    eval_tasks: List[str],
    eval_batches: int,
    seed: int,
) -> Dict[str, object]:
    generator = torch.Generator().manual_seed(seed + 1000 * (stage_index + 1))
    wall_start = time.perf_counter()
    last_loss = 0.0

    for step in range(1, stage_steps + 1):
        model.train()
        x, y, aux = sample_triple_batch(task_name, batch_size, seq_len, device, generator, lookup_table, num_tags=num_tags)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        main_loss = F.cross_entropy(logits, y)
        features = model.encode(x)
        aux_logits = {name: head(features) for name, head in aux_heads.items()}
        aux_loss = (
            F.cross_entropy(aux_logits["a"], aux[:, 0])
            + F.cross_entropy(aux_logits["b"], aux[:, 1])
            + F.cross_entropy(aux_logits["c"], aux[:, 2])
        ) / 3.0
        loss = main_loss + aux_loss_weight * aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(aux_heads.parameters(), 1.0)
        optimizer.step()
        last_loss = loss.item()

        if step % max(stage_steps // 4, 1) == 0:
            print(
                f"[STAGED] {model_name} stage={stage_index + 1:02d} task={task_name} "
                f"step={step:04d}/{stage_steps} loss={loss.item():.4f} "
                f"main={main_loss.item():.4f} aux={aux_loss.item():.4f}"
            )

    duration = time.perf_counter() - wall_start
    metrics = evaluate_all_tasks(
        model=model,
        aux_heads=aux_heads,
        eval_tasks=eval_tasks,
        device=device,
        seq_len=seq_len,
        batch_size=batch_size,
        eval_batches=eval_batches,
        seed=seed + 20_000 + stage_index,
        lookup_table=lookup_table,
        num_tags=num_tags,
    )
    result: Dict[str, object] = {
        "task": "higher_order_staged",
        "protocol": "single_task_checkpointed_continuation",
        "model": model_name,
        "stage_index": stage_index + 1,
        "stage_task": task_name,
        "stage_steps": stage_steps,
        "cumulative_steps": cumulative_steps + stage_steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "seed": seed,
        "device": str(device),
        "num_tags": num_tags,
        "eval_tasks": eval_tasks,
        "train_seconds": duration,
        "train_loss_last": last_loss,
        "aux_loss_weight": aux_loss_weight,
        "num_params": int(model.get_num_params()),
    }
    result.update(metrics)
    return result


def run_single_model(
    *,
    model_name: str,
    root: Path,
    stage_tasks: List[str],
    stage_steps: List[int],
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    lr: float,
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    num_slots: int,
    dropout: float,
    seed: int,
    device: torch.device,
    eval_tasks: List[str],
    num_tags: int,
    aux_loss_weight: float,
    resume: bool,
) -> List[Dict[str, object]]:
    model_root = root / model_name / f"aux_{_weight_token(aux_loss_weight)}"
    checkpoint_dir = model_root / "checkpoints"
    stage_dir = model_root / "stage_results"
    stage_dir.mkdir(parents=True, exist_ok=True)
    lookup_table = build_lookup_table(TRIPLE_NUM_VALUES, TRIPLE_NUM_CLASSES, seed + 101, device)
    cfg = HigherOrderConfig(
        vocab_size=TRIPLE_VOCAB_SIZE,
        num_classes=TRIPLE_NUM_CLASSES,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
        num_slots=num_slots,
        dropout=dropout,
        max_seq_len=seq_len,
        variant_name=model_name,
        query_token_id=TRIPLE_QUERY,
        role_a_token_id=TRIPLE_ROLE_A,
        role_b_token_id=TRIPLE_ROLE_B,
        role_c_token_id=TRIPLE_ROLE_C,
        a_start=TRIPLE_A_START,
        b_start=TRIPLE_B_START,
        c_start=TRIPLE_C_START,
        tag_start=TRIPLE_TAG_START,
        task_start=TRIPLE_TASK_START,
        num_tags=num_tags,
        num_values=TRIPLE_NUM_VALUES,
        num_tasks=TRIPLE_NUM_TASKS,
    )

    existing_rows = load_existing_stage_rows(stage_dir, stage_tasks) if resume else []
    latest_completed_stage = len(existing_rows) - 1

    if latest_completed_stage >= 0:
        checkpoint_path = checkpoint_dir / _stage_checkpoint_name(latest_completed_stage, stage_tasks[latest_completed_stage])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Expected checkpoint missing for resume: {checkpoint_path}")
        loaded_name, loaded_cfg, model, aux_heads, optimizer, _ = load_stage_checkpoint(
            checkpoint_path,
            device=device,
            lr=lr,
        )
        if loaded_name != model_name:
            raise ValueError(f"Resume checkpoint model mismatch: {loaded_name} vs {model_name}")
        if loaded_cfg != cfg:
            raise ValueError(f"Resume checkpoint config mismatch for {model_name}.")
        stage_rows = existing_rows[:]
        cumulative_steps = int(stage_rows[-1]["cumulative_steps"])
    else:
        model = build_higher_order_model(model_name, cfg).to(device)
        aux_heads = build_aux_heads(cfg).to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(aux_heads.parameters()), lr=lr, weight_decay=0.01)
        stage_rows = []
        cumulative_steps = 0

    for stage_index in range(latest_completed_stage + 1, len(stage_tasks)):
        task_name = stage_tasks[stage_index]
        result = train_stage(
            model_name=model_name,
            model=model,
            aux_heads=aux_heads,
            optimizer=optimizer,
            task_name=task_name,
            stage_index=stage_index,
            stage_steps=stage_steps[stage_index],
            cumulative_steps=cumulative_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            lookup_table=lookup_table,
            num_tags=num_tags,
            aux_loss_weight=aux_loss_weight,
            eval_tasks=eval_tasks,
            eval_batches=eval_batches,
            seed=seed,
        )
        checkpoint_path = checkpoint_dir / _stage_checkpoint_name(stage_index, task_name)
        save_stage_checkpoint(
            checkpoint_path,
            model_name=model_name,
            cfg=cfg,
            model=model,
            aux_heads=aux_heads,
            optimizer=optimizer,
            metadata={
                "stage_index": stage_index + 1,
                "stage_task": task_name,
                "cumulative_steps": result["cumulative_steps"],
                "aux_loss_weight": aux_loss_weight,
            },
        )
        result["checkpoint_path"] = str(checkpoint_path)
        (stage_dir / _stage_result_name(stage_index, task_name)).write_text(json.dumps(result, indent=2), encoding="utf-8")
        stage_rows.append(result)
        cumulative_steps = int(result["cumulative_steps"])

    return stage_rows


def run_staged_suite(
    *,
    output_root: str | Path,
    models: List[str],
    stage_tasks: List[str],
    stage_steps: List[int],
    batch_size: int,
    seq_len: int,
    bench_seq_lens: List[int],
    eval_batches: int,
    lr: float,
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    num_slots: int,
    dropout: float,
    seed: int,
    eval_tasks: List[str],
    num_tags: int,
    aux_loss_weight: float,
    resume: bool,
) -> Dict[str, object]:
    device = default_device()
    set_seed(seed)
    root = Path(output_root)
    normalized_steps = _normalize_stage_steps(stage_tasks, stage_steps)

    all_stage_rows: List[Dict[str, object]] = []
    final_rows: List[Dict[str, object]] = []
    for model_name in models:
        stage_rows = run_single_model(
            model_name=model_name,
            root=root / "training",
            stage_tasks=stage_tasks,
            stage_steps=normalized_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            eval_batches=eval_batches,
            lr=lr,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            state_dim=state_dim,
            num_slots=num_slots,
            dropout=dropout,
            seed=seed,
            device=device,
            eval_tasks=eval_tasks,
            num_tags=num_tags,
            aux_loss_weight=aux_loss_weight,
            resume=resume,
        )
        all_stage_rows.extend(stage_rows)
        if stage_rows:
            final_row = dict(stage_rows[-1])
            final_row["task"] = "higher_order_staged_final"
            final_row["train_seconds_last_stage"] = float(stage_rows[-1]["train_seconds"])
            final_row["train_seconds"] = sum(float(row["train_seconds"]) for row in stage_rows)
            final_rows.append(final_row)

    benchmark_rows = benchmark_higher_order_models(
        models=models,
        output_dir=root / "benchmark",
        seq_lens=bench_seq_lens,
        batch_size=batch_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
        num_slots=num_slots,
        dropout=dropout,
        seed=seed + 7,
        device=device,
    )
    training_vs_baseline, benchmark_vs_baseline = build_standard_attention_comparison(
        training_rows=final_rows,
        benchmark_rows=benchmark_rows,
        eval_tasks=eval_tasks,
    )

    write_csv(root / "stage_results.csv", all_stage_rows)
    write_csv(root / "final_results.csv", final_rows)
    write_csv(root / "benchmark_results.csv", benchmark_rows)
    write_csv(root / "final_vs_standard_attention.csv", training_vs_baseline)
    write_csv(root / "benchmark_vs_standard_attention.csv", benchmark_vs_baseline)

    summary = {
        "device": str(device),
        "models": models,
        "stage_tasks": stage_tasks,
        "stage_steps": normalized_steps,
        "eval_tasks": eval_tasks,
        "num_tags": num_tags,
        "aux_loss_weight": aux_loss_weight,
        "resume": resume,
        "standard_attention_baseline": STANDARD_ATTENTION_MODEL if STANDARD_ATTENTION_MODEL in models else None,
        "stage_results": all_stage_rows,
        "final_results": final_rows,
        "benchmark_results": benchmark_rows,
        "final_vs_standard_attention": training_vs_baseline,
        "benchmark_vs_standard_attention": benchmark_vs_baseline,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged higher-order training with checkpointed task progression.")
    parser.add_argument("--output-root", default="auto_research_llm_ideas/results/higher_order_staged")
    parser.add_argument("--models", nargs="+", default=[STANDARD_ATTENTION_MODEL, "pair-state", "hybrid-pair"], choices=HIGHER_ORDER_MODELS)
    parser.add_argument("--stage-tasks", nargs="+", default=TRAIN_TASKS, choices=TRAIN_TASKS)
    parser.add_argument("--stage-steps", nargs="+", type=int, default=[120])
    parser.add_argument("--eval-tasks", nargs="+", default=TRAIN_TASKS, choices=TRAIN_TASKS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--bench-seq-lens", nargs="+", type=int, default=[16, 24, 32])
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--num-slots", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--num-tags", type=int, default=TRIPLE_NUM_TAGS)
    parser.add_argument("--aux-loss-weight", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    summary = run_staged_suite(
        output_root=args.output_root,
        models=args.models,
        stage_tasks=args.stage_tasks,
        stage_steps=args.stage_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        bench_seq_lens=args.bench_seq_lens,
        eval_batches=args.eval_batches,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        state_dim=args.state_dim,
        num_slots=args.num_slots,
        dropout=args.dropout,
        seed=args.seed,
        eval_tasks=args.eval_tasks,
        num_tags=args.num_tags,
        aux_loss_weight=args.aux_loss_weight,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
