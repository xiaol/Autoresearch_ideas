from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from auto_research_llm_ideas.experiments.data import default_device, set_seed
from auto_research_llm_ideas.experiments.train_lm import GENERIC_LM_MODELS, build_model

AR_PAD = 0
AR_BOS = 1
AR_SEP = 2
AR_QUERY = 3
AR_ANSWER = 4
AR_VOCAB_SIZE = 64


def minimum_memory_seq_len(num_pairs: int) -> int:
    prefix_len = 1 + 3 * num_pairs
    suffix_len = 3
    return prefix_len + suffix_len + 1


def sample_memory_batch(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
    num_pairs: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_choices = torch.arange(5, 13)
    value_choices = torch.arange(13, 21)

    inputs = []
    targets = []
    for _ in range(batch_size):
        key_perm = key_choices[torch.randperm(key_choices.numel(), generator=generator)[:num_pairs]]
        value_perm = value_choices[torch.randperm(value_choices.numel(), generator=generator)[:num_pairs]]
        query_index = int(torch.randint(0, num_pairs, (1,), generator=generator).item())

        prefix = [AR_BOS]
        for key, value in zip(key_perm.tolist(), value_perm.tolist()):
            prefix.extend([key, value, AR_SEP])

        suffix = [AR_QUERY, int(key_perm[query_index].item()), AR_ANSWER]
        filler_len = seq_len - len(prefix) - len(suffix)
        if filler_len < 1:
            raise ValueError("Sequence length is too short for the requested associative-recall task.")

        filler = torch.randint(21, 31, (filler_len,), generator=generator).tolist()
        full_sequence = prefix + filler + suffix + [int(value_perm[query_index].item())]
        x = torch.tensor(full_sequence[:-1], dtype=torch.long)
        y = torch.tensor(full_sequence[1:], dtype=torch.long)
        inputs.append(x)
        targets.append(y)

    return torch.stack(inputs, dim=0).to(device), torch.stack(targets, dim=0).to(device)


def curriculum_seq_len(
    *,
    step: int,
    total_steps: int,
    target_seq_len: int,
    curriculum_start_seq_len: int | None,
    curriculum_steps: int,
    num_pairs: int,
) -> int:
    minimum_seq_len = minimum_memory_seq_len(num_pairs)
    start_seq_len = max(minimum_seq_len, curriculum_start_seq_len or target_seq_len)
    if curriculum_steps <= 0 or start_seq_len >= target_seq_len:
        return target_seq_len

    progress = min(max(step - 1, 0), curriculum_steps - 1) / max(curriculum_steps - 1, 1)
    current_seq_len = round(start_seq_len + progress * (target_seq_len - start_seq_len))
    return max(minimum_seq_len, min(target_seq_len, current_seq_len))


def curriculum_num_pairs(
    *,
    step: int,
    target_num_pairs: int,
    curriculum_start_num_pairs: int | None,
    curriculum_steps: int,
) -> int:
    start_num_pairs = max(1, min(target_num_pairs, curriculum_start_num_pairs or target_num_pairs))
    if curriculum_steps <= 0 or start_num_pairs >= target_num_pairs:
        return target_num_pairs

    progress = min(max(step - 1, 0), curriculum_steps - 1) / max(curriculum_steps - 1, 1)
    current_num_pairs = round(start_num_pairs + progress * (target_num_pairs - start_num_pairs))
    return max(1, min(target_num_pairs, current_num_pairs))


def initialize_assoc_gate_biases(
    *,
    model: torch.nn.Module,
    read_gate_bias_init: float | None,
    write_gate_bias_init: float | None,
) -> None:
    if read_gate_bias_init is None and write_gate_bias_init is None:
        return

    for module in model.modules():
        read_gate = getattr(module, "assoc_read_gate_proj", None)
        if read_gate is not None and read_gate.bias is not None and read_gate_bias_init is not None:
            torch.nn.init.constant_(read_gate.bias, read_gate_bias_init)
        write_gate = getattr(module, "assoc_write_gate_proj", None)
        if write_gate is not None and write_gate.bias is not None and write_gate_bias_init is not None:
            torch.nn.init.constant_(write_gate.bias, write_gate_bias_init)


def evaluate_memory_task(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    batches: int,
    seed: int,
    num_pairs: int = 4,
) -> Dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    total_correct = 0
    total_examples = 0
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(batches):
            x, y = sample_memory_batch(batch_size, seq_len, device, generator, num_pairs=num_pairs)
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


def train_memory_model(
    model_name: str,
    output_dir: str | Path,
    steps: int = 250,
    batch_size: int = 32,
    seq_len: int = 128,
    eval_batches: int = 64,
    lr: float = 3e-4,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    state_dim: int = 32,
    dropout: float = 0.1,
    seed: int = 19,
    device: torch.device | None = None,
    num_pairs: int = 4,
    lr_warmup_steps: int = 0,
    curriculum_start_seq_len: int | None = None,
    curriculum_steps: int = 0,
    curriculum_start_num_pairs: int | None = None,
    read_gate_bias_init: float | None = None,
    write_gate_bias_init: float | None = None,
) -> Dict[str, object]:
    device = device or default_device()
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    model = build_model(
        model_name=model_name,
        vocab_size=AR_VOCAB_SIZE,
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
        dropout=dropout,
        pad_token_id=AR_PAD,
    ).to(device)
    initialize_assoc_gate_biases(
        model=model,
        read_gate_bias_init=read_gate_bias_init,
        write_gate_bias_init=write_gate_bias_init,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = None
    if lr_warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step_index: min(float(step_index + 1) / float(lr_warmup_steps), 1.0),
        )
    wall_start = time.perf_counter()
    last_loss = 0.0

    for step in range(1, steps + 1):
        model.train()
        current_seq_len = curriculum_seq_len(
            step=step,
            total_steps=steps,
            target_seq_len=seq_len,
            curriculum_start_seq_len=curriculum_start_seq_len,
            curriculum_steps=curriculum_steps,
            num_pairs=num_pairs,
        )
        current_num_pairs = curriculum_num_pairs(
            step=step,
            target_num_pairs=num_pairs,
            curriculum_start_num_pairs=curriculum_start_num_pairs,
            curriculum_steps=curriculum_steps,
        )
        current_seq_len = max(current_seq_len, minimum_memory_seq_len(current_num_pairs))
        x, y = sample_memory_batch(batch_size, current_seq_len, device, generator, num_pairs=current_num_pairs)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        final_logits = logits[:, -1, :]
        final_targets = y[:, -1]
        loss = F.cross_entropy(final_logits, final_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        last_loss = loss.item()

        if step % max(steps // 5, 1) == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[MEM] {model_name} step {step:04d}/{steps} "
                f"pairs={current_num_pairs} seq={current_seq_len} "
                f"lr={current_lr:.6f} loss={loss.item():.4f}"
            )

    duration = time.perf_counter() - wall_start
    metrics = evaluate_memory_task(
        model,
        device,
        seq_len,
        batch_size,
        eval_batches,
        seed + 1,
        num_pairs=num_pairs,
    )
    result = {
        "task": "associative_recall",
        "model": model_name,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "seed": seed,
        "device": str(device),
        "train_seconds": duration,
        "train_loss_last": last_loss,
        "eval_loss": metrics["loss"],
        "eval_accuracy": metrics["accuracy"],
        "num_params": int(model.get_num_params()),
        "num_pairs": num_pairs,
        "lr_warmup_steps": lr_warmup_steps,
        "curriculum_start_seq_len": curriculum_start_seq_len,
        "curriculum_steps": curriculum_steps,
        "curriculum_start_num_pairs": curriculum_start_num_pairs,
        "read_gate_bias_init": read_gate_bias_init,
        "write_gate_bias_init": write_gate_bias_init,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_memory.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a UniMatrix model on associative recall.")
    parser.add_argument("--model", required=True, choices=GENERIC_LM_MODELS)
    parser.add_argument("--output-dir", default="auto_research_llm_ideas/results/memory")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--num-pairs", type=int, default=4)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--curriculum-start-seq-len", type=int, default=None)
    parser.add_argument("--curriculum-steps", type=int, default=0)
    parser.add_argument("--curriculum-start-num-pairs", type=int, default=None)
    parser.add_argument("--read-gate-bias-init", type=float, default=None)
    parser.add_argument("--write-gate-bias-init", type=float, default=None)
    args = parser.parse_args()

    result = train_memory_model(
        model_name=args.model,
        output_dir=args.output_dir,
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
        num_pairs=args.num_pairs,
        lr_warmup_steps=args.lr_warmup_steps,
        curriculum_start_seq_len=args.curriculum_start_seq_len,
        curriculum_steps=args.curriculum_steps,
        curriculum_start_num_pairs=args.curriculum_start_num_pairs,
        read_gate_bias_init=args.read_gate_bias_init,
        write_gate_bias_init=args.write_gate_bias_init,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
