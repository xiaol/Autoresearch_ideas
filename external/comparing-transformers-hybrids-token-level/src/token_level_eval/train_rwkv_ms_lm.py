from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from token_level_eval.common import parse_dtype
from token_level_eval.score_rwkv_ms import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DELTA_MEM_ROOT,
    DEFAULT_MEMORY_DIR,
    _import_delta_mem,
)


@dataclass
class TrainExample:
    source_id: str
    domain: str
    source: str
    input_ids: list[int]


class TokenWindowDataset(Dataset):
    def __init__(self, examples: list[TrainExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainExample:
        return self.examples[index]


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(model_path: str, *, dtype: str, device: str, attn_implementation: str | None):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "torch_dtype": parse_dtype(dtype),
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(device)


def _read_jsonl_windows(
    path: str | Path,
    tokenizer,
    *,
    max_length: int,
    stride: int,
    limit_records: int | None,
    limit_windows: int | None,
    add_eos: bool,
) -> list[TrainExample]:
    examples: list[TrainExample] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for record_index, line in enumerate(handle):
            if limit_records is not None and record_index >= limit_records:
                break
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if add_eos and tokenizer.eos_token_id is not None:
                token_ids.append(int(tokenizer.eos_token_id))
            if len(token_ids) < 2:
                continue
            step = max_length if stride <= 0 else stride
            for chunk_index, start in enumerate(range(0, len(token_ids), step)):
                chunk = token_ids[start : start + max_length]
                if len(chunk) < 2:
                    continue
                examples.append(
                    TrainExample(
                        source_id=f"{row.get('id', record_index)}:{chunk_index}",
                        domain=str(row.get("domain", "unknown")),
                        source=str(row.get("source", "unknown")),
                        input_ids=chunk,
                    )
                )
                if limit_windows is not None and len(examples) >= limit_windows:
                    return examples
    return examples


def _collate(batch: list[TrainExample], *, pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(example.input_ids) for example in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for row, example in enumerate(batch):
        ids = torch.tensor(example.input_ids, dtype=torch.long)
        input_ids[row, : ids.numel()] = ids
        attention_mask[row, : ids.numel()] = 1
    labels = input_ids.clone()
    labels[attention_mask.eq(0)] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "metadata": [asdict(example) | {"input_ids": None} for example in batch],
    }


def _masked_next_token_loss(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    shift_mask = shift_labels.ne(-100) & attention_mask[:, 1:].ne(0)
    if not shift_mask.any():
        return logits.new_zeros(())
    return F.cross_entropy(shift_logits[shift_mask], shift_labels[shift_mask], reduction="mean")


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        norm = float(param.grad.detach().float().norm().cpu())
        total += norm * norm
    return math.sqrt(total)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a frozen-Gemma4 + RWKV-MS delta-Mem adapter on raw next-token JSONL text."
    )
    parser.add_argument("--delta-mem-root", default=DEFAULT_DELTA_MEM_ROOT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-adapter-dir", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--validation-limit-windows", type=int, default=256)
    parser.add_argument("--add-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--delta-heads", default="q,o")
    parser.add_argument("--target-layers", default="0,1,2,3,4,5")
    parser.add_argument("--online-gain", type=float, default=0.2)
    parser.add_argument("--beta-bias-init", type=float, default=0.0)
    parser.add_argument("--rwkv-ms-num-states", type=int, default=4)
    parser.add_argument("--rwkv-ms-chunk-size", type=int, default=1024)
    return parser


def _evaluate(model, data_loader: DataLoader, *, device: str, reset_fn, max_batches: int | None = None) -> float:
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            reset_fn()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            loss = _masked_next_token_loss(outputs.logits, labels, attention_mask)
            losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    delta_mem_api = _import_delta_mem(args.delta_mem_root)
    from deltamem.core.delta import freeze_non_delta_mem_params, save_delta_mem_adapter
    from deltamem.train.delta_sft_experimental import parse_delta_heads, parse_layer_indices

    tokenizer = _load_tokenizer(args.base_model)
    train_examples = _read_jsonl_windows(
        args.train_jsonl,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        limit_records=args.limit_records,
        limit_windows=args.limit_windows,
        add_eos=args.add_eos,
    )
    if not train_examples:
        raise ValueError("no train token windows were created")
    validation_examples: list[TrainExample] = []
    if args.validation_jsonl:
        validation_examples = _read_jsonl_windows(
            args.validation_jsonl,
            tokenizer,
            max_length=args.max_length,
            stride=args.stride,
            limit_records=None,
            limit_windows=args.validation_limit_windows,
            add_eos=args.add_eos,
        )
    generator = torch.Generator().manual_seed(args.seed)
    collate_fn = lambda batch: _collate(batch, pad_token_id=int(tokenizer.pad_token_id))
    train_loader = DataLoader(
        TokenWindowDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
    )
    validation_loader = (
        DataLoader(
            TokenWindowDataset(validation_examples),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        if validation_examples
        else None
    )

    model = _load_model(
        args.base_model,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )
    config = delta_mem_api["HFDeltaMemConfig"](
        rank=args.rank,
        alpha=args.alpha,
        memory_backend="rwkv_ms",
        target_layers=parse_layer_indices(args.target_layers),
        delta_heads=parse_delta_heads(args.delta_heads),
        beta_bias_init=args.beta_bias_init,
        normalize_qk=True,
        couple_lambda=True,
        state_update_mode="standard",
        rankwise_gates=True,
        output_init="base_slice_fixed",
        base_slice_ref_width=8,
        online_gain=args.online_gain,
        memory_readout_mode="delta",
        memory_write_source="learned_hidden",
        memory_write_granularity="token",
        rwkv_ms_num_states=args.rwkv_ms_num_states,
        rwkv_ms_chunk_size=args.rwkv_ms_chunk_size,
        rwkv_ms_boundary_mode="fixed_chunk",
        rwkv_ms_erase_gate=1.0,
        rwkv_ms_read_top_k=0,
    )
    replaced = delta_mem_api["attach_delta_mem"](model, config)
    if args.resume_adapter_dir:
        delta_mem_api["load_delta_mem_adapter"](model, args.resume_adapter_dir)
    trainable_names = freeze_non_delta_mem_params(model)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    def reset_online_memory() -> None:
        delta_mem_api["reset_delta_mem_states"](model)

    history: list[dict[str, Any]] = []
    step = 0
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, desc="rwkv-ms-lm-train")
    while step < args.max_steps:
        for batch in train_loader:
            step += 1
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)
            reset_online_memory()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            loss = _masked_next_token_loss(outputs.logits, labels, attention_mask)
            (loss / args.gradient_accumulation_steps).backward()
            grad_norm = 0.0
            if step % args.gradient_accumulation_steps == 0:
                grad_norm = _grad_norm(trainable_params)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
            record = {
                "step": step,
                "optimizer_steps": optimizer_steps,
                "loss": float(loss.detach().float().cpu()),
                "grad_norm": grad_norm,
            }
            if validation_loader is not None and args.eval_steps > 0 and step % args.eval_steps == 0:
                record["validation_loss"] = _evaluate(
                    model,
                    validation_loader,
                    device=args.device,
                    reset_fn=reset_online_memory,
                )
            history.append(record)
            if args.logging_steps > 0 and step % args.logging_steps == 0:
                print(json.dumps(record), flush=True)
            if args.save_steps > 0 and step % args.save_steps == 0:
                save_delta_mem_adapter(model, output_dir / "checkpoints" / f"step-{step}", config)
            progress.update(1)
            if step >= args.max_steps:
                break
    progress.close()
    if step > 0 and step % args.gradient_accumulation_steps != 0:
        grad_norm = _grad_norm(trainable_params)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        optimizer_steps += 1
        optimizer.zero_grad(set_to_none=True)
        history[-1]["grad_norm"] = grad_norm
        history[-1]["optimizer_steps"] = optimizer_steps
        history[-1]["final_partial_optimizer_step"] = True

    save_delta_mem_adapter(model, output_dir, config)
    summary = {
        "output_dir": str(output_dir),
        "base_model": args.base_model,
        "train_jsonl": args.train_jsonl,
        "validation_jsonl": args.validation_jsonl,
        "resume_adapter_dir": args.resume_adapter_dir,
        "train_windows": len(train_examples),
        "validation_windows": len(validation_examples),
        "steps": step,
        "optimizer_steps": optimizer_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "num_replaced_modules": len(replaced),
        "first_replaced_modules": replaced[:8],
        "num_trainable_tensors": len(trainable_names),
        "first_trainable_tensors": trainable_names[:8],
        "config": config.to_dict(),
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
