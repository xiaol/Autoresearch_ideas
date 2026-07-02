from __future__ import annotations

import argparse
import json
from pathlib import Path

from token_level_eval.common import parse_dtype, set_seed
from token_level_eval.score_rwkv_ms import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DELTA_MEM_ROOT,
    DEFAULT_MEMORY_DIR,
    _import_delta_mem,
    _load_base_model,
    _load_rwkv_ms_model,
    _load_tokenizer,
    _resolve_device,
    _clear_memory,
)
from token_level_eval.synthetic import evaluate_loaded_model, generate_examples, summarize_probe_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run synthetic probes for local Gemma4 base vs Gemma4 + RWKV-MS online memory."
    )
    parser.add_argument("--delta-mem-root", default=DEFAULT_DELTA_MEM_ROOT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--distances", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    local_files_only = not args.allow_downloads
    device = _resolve_device(args.device)
    delta_mem_api = _import_delta_mem(args.delta_mem_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = generate_examples(args.distances, args.num_examples, args.seed)
    tokenizer = _load_tokenizer(args.base_model, local_files_only=local_files_only)

    print(f"[rwkv-ms] synthetic base model: {args.base_model}", flush=True)
    base_model = _load_base_model(
        model_path=args.base_model,
        device=device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=local_files_only,
        resolve_attn_implementation=delta_mem_api["resolve_attn_implementation"],
    )
    rows = evaluate_loaded_model(base_model, tokenizer, "base", examples)
    del base_model
    _clear_memory()

    print(f"[rwkv-ms] synthetic RWKV-MS adapter: {args.memory_dir}", flush=True)
    rwkv_model, adapter_metadata = _load_rwkv_ms_model(
        model_path=args.base_model,
        memory_dir=args.memory_dir,
        device=device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=local_files_only,
        delta_mem_api=delta_mem_api,
    )

    def reset_online_memory() -> None:
        delta_mem_api["reset_delta_mem_states"](rwkv_model)

    rows.extend(evaluate_loaded_model(rwkv_model, tokenizer, "rwkv_ms_online_memory", examples, reset_fn=reset_online_memory))
    del rwkv_model
    _clear_memory()

    rows_path = output_dir / "synthetic_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summarize_probe_rows(rows).to_csv(output_dir / "synthetic_summary.csv", index=False)
    metadata = {
        "baseline_label": "base",
        "memory_label": "rwkv_ms_online_memory",
        "base_model": args.base_model,
        "memory_dir": str(Path(args.memory_dir).expanduser().resolve()),
        "delta_mem_root": str(Path(args.delta_mem_root).expanduser().resolve()),
        "distances": args.distances,
        "num_examples_per_family_per_distance": args.num_examples,
        "seed": args.seed,
        "dtype": str(parse_dtype(args.dtype)),
        **adapter_metadata,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote synthetic probe rows to {rows_path}")
    print(f"Wrote summary to {output_dir / 'synthetic_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
