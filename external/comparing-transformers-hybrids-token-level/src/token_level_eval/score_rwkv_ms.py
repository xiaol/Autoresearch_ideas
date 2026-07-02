from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Callable

from token_level_eval.common import default_device, load_text_records, parse_dtype, write_jsonl


DEFAULT_DELTA_MEM_ROOT = "/home/xiaol/X/delta-Mem"
DEFAULT_BASE_MODEL = "google/gemma-4-E4B-it"
DEFAULT_MEMORY_DIR = "/home/xiaol/X/hf_gemma_rwkv_step100_upload"


def _resolve_device(device: str) -> str:
    return default_device() if device == "auto" else device


def _import_delta_mem(delta_mem_root: str | Path):
    root = Path(delta_mem_root).expanduser().resolve()
    if not (root / "deltamem").is_dir():
        raise FileNotFoundError(f"{root} does not look like a delta-Mem checkout")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from deltamem.core.delta import (  # type: ignore
        HFDeltaMemConfig,
        attach_delta_mem,
        load_delta_mem_adapter,
        reset_delta_mem_states,
    )
    from deltamem.model_loading import resolve_attn_implementation  # type: ignore

    return {
        "HFDeltaMemConfig": HFDeltaMemConfig,
        "attach_delta_mem": attach_delta_mem,
        "load_delta_mem_adapter": load_delta_mem_adapter,
        "reset_delta_mem_states": reset_delta_mem_states,
        "resolve_attn_implementation": resolve_attn_implementation,
    }


def _load_tokenizer(model_path: str, *, local_files_only: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base_model(
    *,
    model_path: str,
    device: str,
    dtype: str,
    attn_implementation: str | None,
    local_files_only: bool,
    resolve_attn_implementation: Callable[[str, str | None], str | None] | None,
):
    from transformers import AutoModelForCausalLM

    resolved_attn = (
        resolve_attn_implementation(model_path, attn_implementation)
        if resolve_attn_implementation is not None
        else attn_implementation
    )
    kwargs: dict[str, Any] = {
        "torch_dtype": parse_dtype(dtype),
        "device_map": {"": device},
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }
    if resolved_attn is not None:
        kwargs["attn_implementation"] = resolved_attn
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()


def _load_rwkv_ms_model(
    *,
    model_path: str,
    memory_dir: str | Path,
    device: str,
    dtype: str,
    attn_implementation: str | None,
    local_files_only: bool,
    delta_mem_api: dict[str, Any],
):
    model = _load_base_model(
        model_path=model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
        resolve_attn_implementation=delta_mem_api["resolve_attn_implementation"],
    )
    adapter_dir = Path(memory_dir).expanduser().resolve()
    config = delta_mem_api["HFDeltaMemConfig"].from_pretrained(adapter_dir)
    replaced = delta_mem_api["attach_delta_mem"](model, config)
    delta_mem_api["load_delta_mem_adapter"](model, adapter_dir)
    return model.eval(), {
        "adapter_dir": str(adapter_dir),
        "adapter_config": config.to_dict(),
        "replaced_modules": replaced,
    }


def _clear_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unload(model) -> None:
    del model
    _clear_memory()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score token-level gaps for local Gemma4 base vs Gemma4 + "
            "RWKV-MS online-memory delta-Mem adapter."
        )
    )
    parser.add_argument("--delta-mem-root", default=DEFAULT_DELTA_MEM_ROOT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--input", required=True, help="Text file, directory, or JSONL file.")
    parser.add_argument("--domain", default="prose", help="prose, python, html, latex, or auto.")
    parser.add_argument("--jsonl-text-key", default="text")
    parser.add_argument("--jsonl-id-key", default="id")
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--max-copy-ngram", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face downloads. By default this uses local cached model files only.",
    )
    parser.add_argument("--output-jsonl", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from token_level_eval.scoring import encode_with_offsets, paired_token_rows, score_token_nlls

    local_files_only = not args.allow_downloads
    device = _resolve_device(args.device)
    delta_mem_api = _import_delta_mem(args.delta_mem_root)

    records = load_text_records(
        args.input,
        domain=args.domain,
        jsonl_text_key=args.jsonl_text_key,
        jsonl_id_key=args.jsonl_id_key,
        limit=args.limit_records,
    )
    if not records:
        raise ValueError(f"no text records found in {args.input}")

    tokenizer = _load_tokenizer(args.base_model, local_files_only=local_files_only)
    encoded_records = []
    for record in records:
        token_ids, offsets, token_texts = encode_with_offsets(tokenizer, record.text)
        encoded_records.append((record, token_ids, offsets, token_texts))

    print(f"[rwkv-ms] scoring base model: {args.base_model}", flush=True)
    base_model = _load_base_model(
        model_path=args.base_model,
        device=device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=local_files_only,
        resolve_attn_implementation=delta_mem_api["resolve_attn_implementation"],
    )
    base_losses_by_doc: dict[str, dict[int, float]] = {}
    for record, token_ids, _, _ in encoded_records:
        base_losses_by_doc[record.doc_id] = score_token_nlls(
            base_model,
            token_ids,
            seq_len=args.seq_len,
            progress_label=f"base:{record.doc_id}",
        )
    del base_model
    _clear_memory()

    print(f"[rwkv-ms] scoring RWKV-MS adapter: {args.memory_dir}", flush=True)
    rwkv_ms_model, adapter_metadata = _load_rwkv_ms_model(
        model_path=args.base_model,
        memory_dir=args.memory_dir,
        device=device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=local_files_only,
        delta_mem_api=delta_mem_api,
    )

    def reset_online_memory() -> None:
        delta_mem_api["reset_delta_mem_states"](rwkv_ms_model)

    rwkv_losses_by_doc: dict[str, dict[int, float]] = {}
    for record, token_ids, _, _ in encoded_records:
        rwkv_losses_by_doc[record.doc_id] = score_token_nlls(
            rwkv_ms_model,
            token_ids,
            seq_len=args.seq_len,
            progress_label=f"rwkv-ms:{record.doc_id}",
            before_window=reset_online_memory,
        )
    del rwkv_ms_model
    _clear_memory()

    def output_rows():
        for record, token_ids, offsets, token_texts in encoded_records:
            yield from paired_token_rows(
                record,
                tokenizer=tokenizer,
                transformer_losses=base_losses_by_doc[record.doc_id],
                hybrid_losses=rwkv_losses_by_doc[record.doc_id],
                token_ids=token_ids,
                offsets=offsets,
                token_texts=token_texts,
                max_copy_ngram=args.max_copy_ngram,
            )

    write_jsonl(args.output_jsonl, output_rows())

    metadata = {
        "baseline_label": "base",
        "memory_label": "rwkv_ms_online_memory",
        "loss_gap": "base_nll - rwkv_ms_nll; positive means RWKV-MS assigned higher probability",
        "base_model": args.base_model,
        "memory_dir": str(Path(args.memory_dir).expanduser().resolve()),
        "delta_mem_root": str(Path(args.delta_mem_root).expanduser().resolve()),
        "input": str(Path(args.input)),
        "domain": args.domain,
        "seq_len": args.seq_len,
        "max_copy_ngram": args.max_copy_ngram,
        "records": len(records),
        **adapter_metadata,
    }
    meta_path = Path(args.output_jsonl).with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote token rows to {args.output_jsonl}")
    print(f"Wrote metadata to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
