from __future__ import annotations

import argparse
import json
from pathlib import Path

from token_level_eval.common import load_text_records, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score paired token-level NLL gaps for two causal LMs.")
    parser.add_argument("--transformer-model", required=True, help="HF model/path for the transformer baseline.")
    parser.add_argument("--hybrid-model", required=True, help="HF model/path for the hybrid model.")
    parser.add_argument("--tokenizer", default=None, help="Shared tokenizer path. Defaults to --transformer-model.")
    parser.add_argument("--input", required=True, help="Text file, directory, or JSONL file.")
    parser.add_argument("--domain", default="prose", help="prose, python, html, latex, or auto.")
    parser.add_argument("--jsonl-text-key", default="text", help="Text key for JSONL input.")
    parser.add_argument("--jsonl-id-key", default="id", help="ID key for JSONL input.")
    parser.add_argument("--limit-records", type=int, default=None, help="Optional record limit.")
    parser.add_argument("--seq-len", type=int, default=8192, help="Evaluation chunk length.")
    parser.add_argument("--max-copy-ngram", type=int, default=16, help="Repeated n-gram features to emit.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--device", default="auto", help="Device when --device-map is not set.")
    parser.add_argument("--device-map", default=None, help="Optional HF device_map, e.g. auto.")
    parser.add_argument("--attn-implementation", default=None, help="Optional HF attention implementation.")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download HF files.")
    parser.add_argument("--output-jsonl", required=True, help="Output token row JSONL.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from token_level_eval.scoring import ModelLoadConfig, score_records

    records = load_text_records(
        args.input,
        domain=args.domain,
        jsonl_text_key=args.jsonl_text_key,
        jsonl_id_key=args.jsonl_id_key,
        limit=args.limit_records,
    )
    if not records:
        raise ValueError(f"no text records found in {args.input}")

    model_cfg = ModelLoadConfig(
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        trust_remote_code=not args.no_trust_remote_code,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    rows = score_records(
        records,
        transformer_model=args.transformer_model,
        hybrid_model=args.hybrid_model,
        tokenizer_name_or_path=args.tokenizer,
        model_cfg=model_cfg,
        seq_len=args.seq_len,
        max_copy_ngram=args.max_copy_ngram,
    )
    write_jsonl(args.output_jsonl, rows)
    metadata = {
        "transformer_model": args.transformer_model,
        "hybrid_model": args.hybrid_model,
        "tokenizer": args.tokenizer or args.transformer_model,
        "input": str(Path(args.input)),
        "domain": args.domain,
        "seq_len": args.seq_len,
        "max_copy_ngram": args.max_copy_ngram,
        "records": len(records),
    }
    meta_path = Path(args.output_jsonl).with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote token rows to {args.output_jsonl}")
    print(f"Wrote metadata to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
