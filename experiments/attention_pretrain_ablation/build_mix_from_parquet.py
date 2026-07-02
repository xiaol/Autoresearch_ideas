#!/usr/bin/env python3
"""Rebuild the rwkv_ms_hf_mix dataset via direct parquet range-reads.

The canonical builder (prepare_rwkv_ms_hf_dataset.py) uses `datasets` streaming,
which goes through hub endpoints that are currently dropping connections. This
builder reuses the exact same source specs and record-processing helpers, but
pulls rows by HTTP range-reading each source's auto-converted parquet shards
(footer + row groups only), so bandwidth ~ chars consumed, not shard size.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import fsspec
import pyarrow.parquet as pq

CANONICAL = Path(
    "/home/xiaol/X/Autoresearch_ideas/external/comparing-transformers-hybrids-token-level/scripts/prepare_rwkv_ms_hf_dataset.py"
)

spec_mod = importlib.util.spec_from_file_location("prepare_mix", CANONICAL)
mix = importlib.util.module_from_spec(spec_mod)
sys.modules["prepare_mix"] = mix
spec_mod.loader.exec_module(mix)


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.is_file():
        return token_path.read_text().strip()
    return None


def api_json(url: str, token: str | None, retries: int = 5) -> Any:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def iter_parquet_rows(repo: str, config: str, token: str | None) -> Iterator[dict[str, Any]]:
    urls = api_json(f"https://huggingface.co/api/datasets/{repo}/parquet/{config}/train", token)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    fs = fsspec.filesystem("http", headers=headers)
    for url in urls:
        for attempt in range(5):
            try:
                handle = fs.open(url)
                pf = pq.ParquetFile(handle)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        for batch in pf.iter_batches(batch_size=64):
            yield from batch.to_pylist()


def iter_stack_rows(spec, cache_dir: str, token: str | None) -> Iterator[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=spec.repo,
            filename=spec.stack_filename,
            repo_type="dataset",
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception:
        url = f"https://huggingface.co/datasets/{spec.repo}/resolve/main/{spec.stack_filename}"
        dest = Path(cache_dir) / "direct_downloads" / spec.stack_filename.replace("/", "_")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file():
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
        path = str(dest)
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            yield from json.load(handle)
        else:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def iter_rows(spec, cache_dir: str, token: str | None) -> Iterator[dict[str, Any]]:
    if spec.kind == "stack_file":
        yield from iter_stack_rows(spec, cache_dir, token)
    else:
        yield from iter_parquet_rows(spec.repo, spec.config or "default", token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/run/media/xiaol/B214449214445C0B/autoresearch_datasets/rwkv_ms_hf_mix_50mchars")
    parser.add_argument("--hf-cache", default="/run/media/xiaol/B214449214445C0B/hf_cache")
    parser.add_argument("--target-train-chars", type=int, default=50_000_000)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--max-record-chars", type=int, default=12_000)
    parser.add_argument("--max-source-errors", type=int, default=50)
    args = parser.parse_args()

    token = hf_token()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    validation_promille = max(0, min(1000, int(round(args.validation_ratio * 1000))))
    total_weight = sum(s.weight for s in mix.SOURCES)
    source_targets = {s.name: int(args.target_train_chars * s.weight / total_weight) for s in mix.SOURCES}
    source_val_targets = {s.name: int(source_targets[s.name] * args.validation_ratio) for s in mix.SOURCES}

    seen_text_hashes: set[str] = set()
    stats: dict[str, Any] = {
        "train_chars": 0,
        "validation_chars": 0,
        "train_records": 0,
        "validation_records": 0,
        "sources": {},
    }

    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open("w", encoding="utf-8") as val_handle:
        for spec in mix.SOURCES:
            source_train = source_val = source_train_records = source_val_records = 0
            scanned_rows = emitted_docs = errors = 0
            for row in iter_rows(spec, args.hf_cache, token):
                scanned_rows += 1
                try:
                    text = mix.clean_text(row.get(spec.text_field, ""))
                    if len(text) < spec.min_chars:
                        continue
                    source_id = mix.row_source_id(row, spec)
                    doc_key = f"{spec.name}|{source_id}"
                    split = mix.split_for_doc(doc_key, validation_promille)
                    if split == "train" and source_train >= source_targets[spec.name]:
                        continue
                    if split == "validation" and source_val >= source_val_targets[spec.name]:
                        continue
                    emitted_doc = False
                    for chunk_index, chunk in enumerate(
                        mix.iter_text_chunks(text, max_chars=args.max_record_chars, min_chars=spec.min_chars)
                    ):
                        text_hash = mix.stable_hash(chunk)
                        if text_hash in seen_text_hashes:
                            continue
                        seen_text_hashes.add(text_hash)
                        record_id = f"{spec.name}:{mix.stable_hash(source_id)[:12]}:{chunk_index:04d}:{text_hash[:12]}"
                        if split == "validation":
                            if source_val >= source_val_targets[spec.name]:
                                break
                            chars = mix.write_record(val_handle, record_id=record_id, spec=spec, source_id=source_id, chunk_index=chunk_index, text=chunk)
                            source_val += chars
                            source_val_records += 1
                            stats["validation_chars"] += chars
                            stats["validation_records"] += 1
                        else:
                            if source_train >= source_targets[spec.name]:
                                break
                            chars = mix.write_record(train_handle, record_id=record_id, spec=spec, source_id=source_id, chunk_index=chunk_index, text=chunk)
                            source_train += chars
                            source_train_records += 1
                            stats["train_chars"] += chars
                            stats["train_records"] += 1
                        emitted_doc = True
                    if emitted_doc:
                        emitted_docs += 1
                    if source_train >= source_targets[spec.name] and source_val >= source_val_targets[spec.name]:
                        break
                except Exception as exc:  # noqa: BLE001 - keep progress across malformed rows
                    errors += 1
                    if errors > args.max_source_errors:
                        raise RuntimeError(f"too many errors in {spec.name}") from exc
            stats["sources"][spec.name] = {
                "domain": spec.domain,
                "hf_repo": spec.repo,
                "hf_config": spec.config,
                "kind": spec.kind,
                "target_train_chars": source_targets[spec.name],
                "target_validation_chars": source_val_targets[spec.name],
                "train_chars": source_train,
                "validation_chars": source_val,
                "train_records": source_train_records,
                "validation_records": source_val_records,
                "scanned_rows": scanned_rows,
                "emitted_docs": emitted_docs,
                "errors": errors,
            }
            print(
                f"{spec.name}: train_chars={source_train} validation_chars={source_val} "
                f"train_records={source_train_records} validation_records={source_val_records}",
                flush=True,
            )

    stats.update(
        {
            "schema": "rwkv_ms_hf_mix.v1",
            "built_via": "parquet_range_reader",
            "train_path": str(train_path),
            "validation_path": str(val_path),
            "hf_cache": args.hf_cache,
            "target_train_chars": args.target_train_chars,
            "validation_ratio": args.validation_ratio,
            "max_record_chars": args.max_record_chars,
            "estimated_train_tokens_chars_div_4": stats["train_chars"] // 4,
            "estimated_validation_tokens_chars_div_4": stats["validation_chars"] // 4,
        }
    )
    (output_dir / "manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
