from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F

from auto_research_llm_ideas.experiments.data import default_device, set_seed
from auto_research_llm_ideas.model.higher_order import HIGHER_ORDER_MODELS, HigherOrderConfig, build_higher_order_model

TRIPLE_PAD = 0
TRIPLE_BOS = 1
TRIPLE_QUERY = 2
TRIPLE_ROLE_A = 3
TRIPLE_ROLE_B = 4
TRIPLE_ROLE_C = 5
TRIPLE_FILLER_START = 6
TRIPLE_FILLER_COUNT = 16
TRIPLE_A_START = TRIPLE_FILLER_START + TRIPLE_FILLER_COUNT
TRIPLE_B_START = TRIPLE_A_START + 8
TRIPLE_C_START = TRIPLE_B_START + 8
TRIPLE_TAG_START = TRIPLE_C_START + 8
TRIPLE_NUM_TAGS = 3
TRIPLE_NUM_VALUES = 8
TRIPLE_NUM_CLASSES = 16
TRAIN_TASKS = ["binding-copy", "binding-affine", "binding-gate", "binding-lookup"]
STANDARD_ATTENTION_MODEL = "transformer-triple"
TRIPLE_TASK_START = TRIPLE_TAG_START + TRIPLE_NUM_TAGS
TASK_TOKEN_IDS = {task_name: TRIPLE_TASK_START + index for index, task_name in enumerate(TRAIN_TASKS)}
TRIPLE_NUM_TASKS = len(TRAIN_TASKS)
TRIPLE_VOCAB_SIZE = TRIPLE_TASK_START + TRIPLE_NUM_TASKS


def copy_target(a_idx: int, b_idx: int, c_idx: int) -> int:
    del b_idx, c_idx
    return a_idx


def affine_target(a_idx: int, b_idx: int, c_idx: int) -> int:
    return (a_idx + 2 * b_idx + 3 * c_idx) % TRIPLE_NUM_CLASSES


def gate_target(a_idx: int, b_idx: int, c_idx: int) -> int:
    return ((a_idx * (b_idx + 1)) + c_idx * ((a_idx ^ b_idx) + 1)) % TRIPLE_NUM_CLASSES


TASK_FNS: Dict[str, Callable[[int, int, int], int]] = {
    "binding-copy": copy_target,
    "binding-affine": affine_target,
    "binding-gate": gate_target,
}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def build_standard_attention_comparison(
    training_rows: List[Dict[str, object]],
    benchmark_rows: List[Dict[str, object]],
    eval_tasks: List[str],
    baseline_model: str = STANDARD_ATTENTION_MODEL,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    train_by_model = {str(row["model"]): row for row in training_rows}
    baseline_train = train_by_model.get(baseline_model)
    if baseline_train is None:
        return [], []

    training_comparison_rows: List[Dict[str, object]] = []
    for model_name, row in train_by_model.items():
        if model_name == baseline_model:
            continue
        comparison = {
            "model": model_name,
            "baseline_model": baseline_model,
            "num_params_ratio_vs_baseline": _safe_ratio(float(row["num_params"]), float(baseline_train["num_params"])),
            "train_seconds_ratio_vs_baseline": _safe_ratio(
                float(row["train_seconds"]),
                float(baseline_train["train_seconds"]),
            ),
        }
        for task_name in eval_tasks:
            safe_name = task_name.replace("-", "_")
            acc_key = f"{safe_name}_accuracy"
            loss_key = f"{safe_name}_loss"
            if acc_key in row and acc_key in baseline_train:
                comparison[f"{acc_key}_delta_vs_baseline"] = float(row[acc_key]) - float(baseline_train[acc_key])
            if loss_key in row and loss_key in baseline_train:
                comparison[f"{loss_key}_delta_vs_baseline"] = float(row[loss_key]) - float(baseline_train[loss_key])
        training_comparison_rows.append(comparison)

    bench_by_key = {(str(row["model"]), int(row["seq_len"])): row for row in benchmark_rows}
    benchmark_comparison_rows: List[Dict[str, object]] = []
    for row in benchmark_rows:
        model_name = str(row["model"])
        if model_name == baseline_model:
            continue
        baseline_row = bench_by_key.get((baseline_model, int(row["seq_len"])))
        if baseline_row is None:
            continue
        benchmark_comparison_rows.append(
            {
                "model": model_name,
                "baseline_model": baseline_model,
                "seq_len": int(row["seq_len"]),
                "num_params_ratio_vs_baseline": _safe_ratio(
                    float(row["num_params"]),
                    float(baseline_row["num_params"]),
                ),
                "examples_per_second_ratio_vs_baseline": _safe_ratio(
                    float(row["examples_per_second"]),
                    float(baseline_row["examples_per_second"]),
                ),
                "milliseconds_per_iter_ratio_vs_baseline": _safe_ratio(
                    float(row["milliseconds_per_iter"]),
                    float(baseline_row["milliseconds_per_iter"]),
                ),
                "memory_mb_ratio_vs_baseline": _safe_ratio(
                    float(row["memory_mb"]),
                    float(baseline_row["memory_mb"]),
                ),
            }
        )
    return training_comparison_rows, benchmark_comparison_rows


def current_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 ** 2)
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def build_lookup_table(num_values: int, num_classes: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    del device
    return torch.randint(0, num_classes, (num_values, num_values, num_values), generator=generator)


def _random_partition(total: int, parts: int, generator: torch.Generator) -> List[int]:
    if total <= 0:
        return [0] * parts
    probs = torch.rand(parts, generator=generator) + 1e-4
    draws = torch.multinomial(probs / probs.sum(), total, replacement=True, generator=generator)
    return torch.bincount(draws, minlength=parts).tolist()


def minimum_seq_len(num_tags: int) -> int:
    return 2 + (3 * num_tags * 3) + 2


def task_target(
    task_name: str,
    a_idx: int,
    b_idx: int,
    c_idx: int,
    lookup_table: torch.Tensor,
) -> int:
    if task_name == "binding-lookup":
        return int(lookup_table[a_idx, b_idx, c_idx].item())
    if task_name in TASK_FNS:
        return int(TASK_FNS[task_name](a_idx, b_idx, c_idx))
    raise ValueError(f"Unknown higher-order task: {task_name}")


def task_uses_shuffled_chunks(task_name: str) -> bool:
    return task_name == "binding-lookup"


def sample_triple_batch(
    task_name: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
    lookup_table: torch.Tensor,
    num_tags: int = TRIPLE_NUM_TAGS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_len = minimum_seq_len(num_tags)
    if seq_len < min_len:
        raise ValueError(
            f"Sequence length must be at least {min_len} for task {task_name} with num_tags={num_tags}."
        )

    filler_choices = torch.arange(TRIPLE_FILLER_START, TRIPLE_FILLER_START + TRIPLE_FILLER_COUNT)
    inputs = []
    targets = []
    aux_targets = []

    for _ in range(batch_size):
        tag_values = []
        chunks = []
        for tag_index in range(num_tags):
            a_idx = int(torch.randint(0, TRIPLE_NUM_VALUES, (1,), generator=generator).item())
            b_idx = int(torch.randint(0, TRIPLE_NUM_VALUES, (1,), generator=generator).item())
            c_idx = int(torch.randint(0, TRIPLE_NUM_VALUES, (1,), generator=generator).item())
            tag_token = TRIPLE_TAG_START + tag_index
            tag_values.append((a_idx, b_idx, c_idx))
            chunks.extend(
                [
                    [tag_token, TRIPLE_ROLE_A, TRIPLE_A_START + a_idx],
                    [tag_token, TRIPLE_ROLE_B, TRIPLE_B_START + b_idx],
                    [tag_token, TRIPLE_ROLE_C, TRIPLE_C_START + c_idx],
                ]
            )

        target_tag = int(torch.randint(0, num_tags, (1,), generator=generator).item())
        a_idx, b_idx, c_idx = tag_values[target_tag]
        target = task_target(task_name, a_idx, b_idx, c_idx, lookup_table)
        sequence = [TRIPLE_BOS, TASK_TOKEN_IDS[task_name]]

        filler_budget = seq_len - min_len
        if task_uses_shuffled_chunks(task_name):
            order = torch.randperm(len(chunks), generator=generator).tolist()
            gap_counts = _random_partition(filler_budget, len(chunks) + 1, generator)
            for gap_index, chunk_index in enumerate(order):
                if gap_counts[gap_index] > 0:
                    filler = filler_choices[
                        torch.randint(0, filler_choices.numel(), (gap_counts[gap_index],), generator=generator)
                    ]
                    sequence.extend(filler.tolist())
                sequence.extend(chunks[chunk_index])
            if gap_counts[-1] > 0:
                filler = filler_choices[
                    torch.randint(0, filler_choices.numel(), (gap_counts[-1],), generator=generator)
                ]
                sequence.extend(filler.tolist())
        else:
            records = []
            for tag_index, (a_val, b_val, c_val) in enumerate(tag_values):
                tag_token = TRIPLE_TAG_START + tag_index
                records.append(
                    [
                        tag_token,
                        TRIPLE_ROLE_A,
                        TRIPLE_A_START + a_val,
                        TRIPLE_ROLE_B,
                        TRIPLE_B_START + b_val,
                        TRIPLE_ROLE_C,
                        TRIPLE_C_START + c_val,
                    ]
                )
            order = torch.randperm(len(records), generator=generator).tolist()
            gap_counts = _random_partition(filler_budget, len(records) + 1, generator)
            for gap_index, record_index in enumerate(order):
                if gap_counts[gap_index] > 0:
                    filler = filler_choices[
                        torch.randint(0, filler_choices.numel(), (gap_counts[gap_index],), generator=generator)
                    ]
                    sequence.extend(filler.tolist())
                sequence.extend(records[record_index])
            if gap_counts[-1] > 0:
                filler = filler_choices[
                    torch.randint(0, filler_choices.numel(), (gap_counts[-1],), generator=generator)
                ]
                sequence.extend(filler.tolist())
        sequence.extend([TRIPLE_QUERY, TRIPLE_TAG_START + target_tag])

        inputs.append(torch.tensor(sequence, dtype=torch.long))
        targets.append(target)
        aux_targets.append([a_idx, b_idx, c_idx])

    return (
        torch.stack(inputs, dim=0).to(device),
        torch.tensor(targets, dtype=torch.long, device=device),
        torch.tensor(aux_targets, dtype=torch.long, device=device),
    )


def evaluate_triple_task(
    model: torch.nn.Module,
    task_name: str,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    batches: int,
    seed: int,
    lookup_table: torch.Tensor,
    num_tags: int = TRIPLE_NUM_TAGS,
    aux_heads: torch.nn.ModuleDict | None = None,
) -> Dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    total_correct = 0
    total_examples = 0
    total_loss = 0.0
    aux_correct = {"a": 0, "b": 0, "c": 0}

    with torch.no_grad():
        for _ in range(batches):
            x, y, aux = sample_triple_batch(task_name, batch_size, seq_len, device, generator, lookup_table, num_tags=num_tags)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_examples += batch_size
            if aux_heads is not None:
                features = model.encode(x)
                aux_logits = {name: head(features) for name, head in aux_heads.items()}
                aux_correct["a"] += (aux_logits["a"].argmax(dim=-1) == aux[:, 0]).sum().item()
                aux_correct["b"] += (aux_logits["b"].argmax(dim=-1) == aux[:, 1]).sum().item()
                aux_correct["c"] += (aux_logits["c"].argmax(dim=-1) == aux[:, 2]).sum().item()

    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }
    if aux_heads is not None:
        metrics["aux_a_accuracy"] = aux_correct["a"] / max(total_examples, 1)
        metrics["aux_b_accuracy"] = aux_correct["b"] / max(total_examples, 1)
        metrics["aux_c_accuracy"] = aux_correct["c"] / max(total_examples, 1)
    return metrics


def curriculum_task(step: int, total_steps: int, train_tasks: List[str]) -> str:
    if len(train_tasks) == 1:
        return train_tasks[0]
    phase = step / max(total_steps, 1)
    phase_index = min(int(phase * len(train_tasks)), len(train_tasks) - 1)
    return train_tasks[phase_index]


def train_triple_model(
    model_name: str,
    output_dir: str | Path,
    steps: int = 300,
    batch_size: int = 32,
    seq_len: int = 64,
    eval_batches: int = 64,
    lr: float = 3e-4,
    d_model: int = 128,
    n_layers: int = 3,
    n_heads: int = 4,
    state_dim: int = 16,
    num_slots: int = 8,
    dropout: float = 0.1,
    seed: int = 23,
    device: torch.device | None = None,
    train_tasks: List[str] | None = None,
    eval_tasks: List[str] | None = None,
    num_tags: int = TRIPLE_NUM_TAGS,
    aux_loss_weight: float = 0.25,
) -> Dict[str, object]:
    device = device or default_device()
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_tasks = train_tasks or TRAIN_TASKS
    eval_tasks = eval_tasks or TRAIN_TASKS
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
    model = build_higher_order_model(model_name, cfg).to(device)
    aux_heads = torch.nn.ModuleDict(
        {
            "a": torch.nn.Linear(cfg.d_model, TRIPLE_NUM_VALUES),
            "b": torch.nn.Linear(cfg.d_model, TRIPLE_NUM_VALUES),
            "c": torch.nn.Linear(cfg.d_model, TRIPLE_NUM_VALUES),
        }
    ).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(aux_heads.parameters()), lr=lr, weight_decay=0.01)
    wall_start = time.perf_counter()
    last_loss = 0.0
    last_task = train_tasks[-1]

    for step in range(1, steps + 1):
        model.train()
        task_name = curriculum_task(step, steps, train_tasks)
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
        last_task = task_name

        if step % max(steps // 5, 1) == 0:
            print(
                f"[TRIPLE] {model_name} step {step:04d}/{steps} task={task_name} "
                f"loss={loss.item():.4f} main={main_loss.item():.4f} aux={aux_loss.item():.4f}"
            )

    duration = time.perf_counter() - wall_start
    eval_metrics = {}
    for task_name in eval_tasks:
        eval_metrics[task_name] = evaluate_triple_task(
            model,
            task_name,
            device,
            seq_len,
            batch_size,
            eval_batches,
            seed + 1,
            lookup_table,
            num_tags=num_tags,
            aux_heads=aux_heads,
        )
    result = {
        "task": "higher_order_curriculum",
        "model": model_name,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "seed": seed,
        "device": str(device),
        "num_tags": num_tags,
        "train_tasks": train_tasks,
        "eval_tasks": eval_tasks,
        "train_seconds": duration,
        "train_loss_last": last_loss,
        "train_task_last": last_task,
        "aux_loss_weight": aux_loss_weight,
        "num_params": int(model.get_num_params()),
    }
    for task_name, metrics in eval_metrics.items():
        safe_name = task_name.replace("-", "_")
        result[f"{safe_name}_loss"] = metrics["loss"]
        result[f"{safe_name}_accuracy"] = metrics["accuracy"]
        if "aux_a_accuracy" in metrics:
            result[f"{safe_name}_aux_a_accuracy"] = metrics["aux_a_accuracy"]
            result[f"{safe_name}_aux_b_accuracy"] = metrics["aux_b_accuracy"]
            result[f"{safe_name}_aux_c_accuracy"] = metrics["aux_c_accuracy"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_triple.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def benchmark_higher_order_models(
    models: List[str],
    output_dir: str | Path,
    seq_lens: List[int],
    batch_size: int = 16,
    d_model: int = 128,
    n_layers: int = 3,
    n_heads: int = 4,
    state_dim: int = 16,
    num_slots: int = 8,
    dropout: float = 0.1,
    warmup_iters: int = 3,
    timed_iters: int = 10,
    seed: int = 31,
    device: torch.device | None = None,
) -> List[Dict[str, object]]:
    device = device or default_device()
    set_seed(seed)
    results: List[Dict[str, object]] = []

    for model_name in models:
        max_seq_len = max(seq_lens)
        cfg = HigherOrderConfig(
            vocab_size=TRIPLE_VOCAB_SIZE,
            num_classes=TRIPLE_NUM_CLASSES,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            state_dim=state_dim,
            num_slots=num_slots,
            dropout=dropout,
            max_seq_len=max_seq_len,
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
            num_tags=TRIPLE_NUM_TAGS,
            num_values=TRIPLE_NUM_VALUES,
            num_tasks=TRIPLE_NUM_TASKS,
        )
        model = build_higher_order_model(model_name, cfg).to(device)
        model.eval()

        for seq_len in seq_lens:
            tokens = torch.randint(0, TRIPLE_VOCAB_SIZE, (batch_size, seq_len), device=device)
            if seq_len >= 2:
                tokens[:, 0] = TRIPLE_BOS
                task_ids = torch.tensor(list(TASK_TOKEN_IDS.values()), device=device)
                task_choices = task_ids[torch.randint(0, task_ids.numel(), (batch_size,), device=device)]
                tokens[:, 1] = task_choices
                tokens[:, -2] = TRIPLE_QUERY
                tokens[:, -1] = TRIPLE_TAG_START + torch.randint(0, TRIPLE_NUM_TAGS, (batch_size,), device=device)
            else:
                tokens[:, -1] = TRIPLE_BOS

            for _ in range(warmup_iters):
                with torch.no_grad():
                    model(tokens)
            synchronize(device)

            start_memory = current_memory_mb(device)
            start = time.perf_counter()
            for _ in range(timed_iters):
                with torch.no_grad():
                    model(tokens)
            synchronize(device)
            elapsed = time.perf_counter() - start
            end_memory = current_memory_mb(device)

            examples_processed = batch_size * timed_iters
            results.append(
                {
                    "task": "triple_interaction_benchmark",
                    "model": model_name,
                    "device": str(device),
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "timed_iters": timed_iters,
                    "num_params": int(model.get_num_params()),
                    "examples_per_second": examples_processed / max(elapsed, 1e-9),
                    "milliseconds_per_iter": (elapsed / timed_iters) * 1000.0,
                    "memory_mb": max(start_memory, end_memory),
                }
            )
            print(
                f"[HOBENCH] {model_name} seq={seq_len:4d} "
                f"ex/s={results[-1]['examples_per_second']:.1f} mem={results[-1]['memory_mb']:.1f}MB"
            )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "higher_order_benchmark.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_suite(
    output_root: str | Path,
    models: List[str],
    steps: int,
    batch_size: int,
    seq_len: int,
    bench_seq_lens: List[int],
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    num_slots: int,
    train_tasks: List[str],
    eval_tasks: List[str],
    num_tags: int,
    aux_loss_weight: float,
) -> Dict[str, object]:
    device = default_device()
    root = Path(output_root)

    training_rows = []
    for model_name in models:
        training_rows.append(
            train_triple_model(
                model_name=model_name,
                output_dir=root / "training",
                steps=steps,
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                state_dim=state_dim,
                num_slots=num_slots,
                device=device,
                train_tasks=train_tasks,
                eval_tasks=eval_tasks,
                num_tags=num_tags,
                aux_loss_weight=aux_loss_weight,
            )
        )

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
        device=device,
    )

    write_csv(root / "training_results.csv", training_rows)
    write_csv(root / "benchmark_results.csv", benchmark_rows)
    training_vs_baseline, benchmark_vs_baseline = build_standard_attention_comparison(
        training_rows=training_rows,
        benchmark_rows=benchmark_rows,
        eval_tasks=eval_tasks,
    )
    write_csv(root / "training_vs_standard_attention.csv", training_vs_baseline)
    write_csv(root / "benchmark_vs_standard_attention.csv", benchmark_vs_baseline)

    summary = {
        "device": str(device),
        "models": models,
        "train_tasks": train_tasks,
        "eval_tasks": eval_tasks,
        "num_tags": num_tags,
        "aux_loss_weight": aux_loss_weight,
        "standard_attention_baseline": STANDARD_ATTENTION_MODEL if STANDARD_ATTENTION_MODEL in models else None,
        "training_results": training_rows,
        "benchmark_results": benchmark_rows,
        "training_vs_standard_attention": training_vs_baseline,
        "benchmark_vs_standard_attention": benchmark_vs_baseline,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the higher-order triple-interaction research suite.")
    parser.add_argument("--output-root", default="auto_research_llm_ideas/results/higher_order")
    parser.add_argument("--models", nargs="+", default=HIGHER_ORDER_MODELS, choices=HIGHER_ORDER_MODELS)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--bench-seq-lens", nargs="+", type=int, default=[32, 48, 64, 80])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--train-tasks", nargs="+", default=TRAIN_TASKS, choices=TRAIN_TASKS)
    parser.add_argument("--eval-tasks", nargs="+", default=TRAIN_TASKS, choices=TRAIN_TASKS)
    parser.add_argument("--num-tags", type=int, default=TRIPLE_NUM_TAGS)
    parser.add_argument("--aux-loss-weight", type=float, default=0.25)
    args = parser.parse_args()

    summary = run_suite(
        output_root=args.output_root,
        models=args.models,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        bench_seq_lens=args.bench_seq_lens,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        state_dim=args.state_dim,
        num_slots=args.num_slots,
        train_tasks=args.train_tasks,
        eval_tasks=args.eval_tasks,
        num_tags=args.num_tags,
        aux_loss_weight=args.aux_loss_weight,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
