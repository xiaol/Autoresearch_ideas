#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_HF_CACHE = "/run/media/xiaol/B214449214445C0B/hf_cache"
DEFAULT_OUTPUT_DIR = "/run/media/xiaol/B214449214445C0B/autoresearch_datasets/rwkv_ms_hf_mix"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    domain: str
    weight: float
    kind: str
    repo: str
    config: str | None = None
    split: str = "train"
    text_field: str = "text"
    id_fields: tuple[str, ...] = ("id",)
    stack_filename: str | None = None
    min_chars: int = 400


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="wikipedia",
        domain="prose",
        weight=0.18,
        kind="dataset_stream",
        repo="wikimedia/wikipedia",
        config="20231101.en",
        text_field="text",
        id_fields=("id", "title"),
    ),
    SourceSpec(
        name="cc_news",
        domain="prose",
        weight=0.17,
        kind="dataset_stream",
        repo="vblagoje/cc_news",
        config="plain_text",
        text_field="text",
        id_fields=("url", "title", "date"),
    ),
    SourceSpec(
        name="arxiv_article",
        domain="prose",
        weight=0.15,
        kind="dataset_stream",
        repo="ccdv/arxiv-summarization",
        config="document",
        text_field="article",
        id_fields=("abstract",),
    ),
    SourceSpec(
        name="python_codesearchnet",
        domain="python",
        weight=0.25,
        kind="dataset_stream",
        repo="code-search-net/code_search_net",
        config="python",
        text_field="whole_func_string",
        id_fields=("repository_name", "func_path_in_repository", "func_name"),
        min_chars=250,
    ),
    SourceSpec(
        name="html_stack_smol_xl",
        domain="html",
        weight=0.10,
        kind="stack_file",
        repo="bigcode/the-stack-smol-xl",
        text_field="content",
        id_fields=("hexsha", "max_stars_repo_path"),
        stack_filename="data/html/data.json",
        min_chars=250,
    ),
    SourceSpec(
        name="latex_arxiv_source",
        domain="latex",
        weight=0.10,
        kind="dataset_stream",
        repo="scholarweave/arxiv-latex",
        config="default",
        text_field="latex",
        id_fields=("id", "title"),
        min_chars=500,
    ),
    SourceSpec(
        name="markdown_stack_smol_xl",
        domain="prose",
        weight=0.05,
        kind="stack_file",
        repo="bigcode/the-stack-smol-xl",
        text_field="content",
        id_fields=("hexsha", "max_stars_repo_path"),
        stack_filename="data/markdown/data.json",
        min_chars=250,
    ),
)


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def row_source_id(row: dict[str, Any], spec: SourceSpec) -> str:
    values = [clean_text(row.get(field, "")) for field in spec.id_fields]
    joined = "|".join(value for value in values if value)
    if joined:
        return joined[:512]
    text = clean_text(row.get(spec.text_field, ""))
    return stable_hash(f"{spec.name}|{text[:2048]}")


def iter_text_chunks(text: str, *, max_chars: int, min_chars: int) -> Iterator[str]:
    text = clean_text(text)
    if len(text) < min_chars:
        return
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        if end < n:
            boundary = text.rfind("\n\n", start + min_chars, end)
            if boundary < 0:
                boundary = text.rfind("\n", start + min_chars, end)
            if boundary < 0:
                boundary = text.rfind(" ", start + min_chars, end)
            if boundary >= 0:
                end = boundary
        chunk = text[start:end].strip()
        if len(chunk) >= min_chars:
            yield chunk
        start = max(end, start + 1)


def iter_dataset_stream(spec: SourceSpec, *, cache_dir: str) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "split": spec.split,
        "streaming": True,
        "cache_dir": cache_dir,
    }
    if spec.config:
        dataset = load_dataset(spec.repo, spec.config, **kwargs)
    else:
        dataset = load_dataset(spec.repo, **kwargs)
    for row in dataset:
        yield row


def iter_stack_file(spec: SourceSpec, *, cache_dir: str) -> Iterator[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    if not spec.stack_filename:
        raise ValueError(f"{spec.name} is missing stack_filename")
    path = hf_hub_download(
        repo_id=spec.repo,
        filename=spec.stack_filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            rows = json.load(handle)
            yield from rows
        else:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def iter_source_rows(spec: SourceSpec, *, cache_dir: str) -> Iterator[dict[str, Any]]:
    if spec.kind == "dataset_stream":
        yield from iter_dataset_stream(spec, cache_dir=cache_dir)
    elif spec.kind == "stack_file":
        yield from iter_stack_file(spec, cache_dir=cache_dir)
    else:
        raise ValueError(f"unknown source kind: {spec.kind}")


def split_for_doc(doc_key: str, validation_promille: int) -> str:
    bucket = int(stable_hash(doc_key)[:8], 16) % 1000
    return "validation" if bucket < validation_promille else "train"


def write_record(handle, *, record_id: str, spec: SourceSpec, source_id: str, chunk_index: int, text: str) -> int:
    row = {
        "id": record_id,
        "domain": spec.domain,
        "source": spec.name,
        "hf_repo": spec.repo,
        "hf_config": spec.config,
        "source_id": source_id,
        "chunk_index": chunk_index,
        "text": text,
    }
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(text)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(Path(args.hf_cache).expanduser())
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    validation_promille = max(0, min(1000, int(round(args.validation_ratio * 1000))))
    total_weight = sum(source.weight for source in SOURCES)
    source_targets = {
        source.name: int(args.target_train_chars * source.weight / total_weight)
        for source in SOURCES
    }
    source_val_targets = {
        source.name: int(source_targets[source.name] * args.validation_ratio)
        for source in SOURCES
    }

    seen_text_hashes: set[str] = set()
    stats: dict[str, Any] = {
        "train_chars": 0,
        "validation_chars": 0,
        "train_records": 0,
        "validation_records": 0,
        "sources": {},
    }

    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open("w", encoding="utf-8") as val_handle:
        for spec in SOURCES:
            source_train = 0
            source_val = 0
            source_train_records = 0
            source_val_records = 0
            scanned_rows = 0
            emitted_docs = 0
            errors = 0
            for row in iter_source_rows(spec, cache_dir=cache_dir):
                scanned_rows += 1
                try:
                    text = clean_text(row.get(spec.text_field, ""))
                    if len(text) < spec.min_chars:
                        continue
                    source_id = row_source_id(row, spec)
                    doc_key = f"{spec.name}|{source_id}"
                    split = split_for_doc(doc_key, validation_promille)
                    if split == "train" and source_train >= source_targets[spec.name]:
                        continue
                    if split == "validation" and source_val >= source_val_targets[spec.name]:
                        continue
                    emitted_doc = False
                    for chunk_index, chunk in enumerate(
                        iter_text_chunks(text, max_chars=args.max_record_chars, min_chars=spec.min_chars)
                    ):
                        text_hash = stable_hash(chunk)
                        if text_hash in seen_text_hashes:
                            continue
                        seen_text_hashes.add(text_hash)
                        record_id = f"{spec.name}:{stable_hash(source_id)[:12]}:{chunk_index:04d}:{text_hash[:12]}"
                        if split == "validation":
                            if source_val >= source_val_targets[spec.name]:
                                break
                            chars = write_record(
                                val_handle,
                                record_id=record_id,
                                spec=spec,
                                source_id=source_id,
                                chunk_index=chunk_index,
                                text=chunk,
                            )
                            source_val += chars
                            source_val_records += 1
                            stats["validation_chars"] += chars
                            stats["validation_records"] += 1
                        else:
                            if source_train >= source_targets[spec.name]:
                                break
                            chars = write_record(
                                train_handle,
                                record_id=record_id,
                                spec=spec,
                                source_id=source_id,
                                chunk_index=chunk_index,
                                text=chunk,
                            )
                            source_train += chars
                            source_train_records += 1
                            stats["train_chars"] += chars
                            stats["train_records"] += 1
                        emitted_doc = True
                    if emitted_doc:
                        emitted_docs += 1
                    if source_train >= source_targets[spec.name] and source_val >= source_val_targets[spec.name]:
                        break
                except Exception as exc:  # noqa: BLE001 - preserve progress across occasional malformed rows.
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
            "train_path": str(train_path),
            "validation_path": str(val_path),
            "hf_cache": cache_dir,
            "target_train_chars": args.target_train_chars,
            "validation_ratio": args.validation_ratio,
            "max_record_chars": args.max_record_chars,
            "estimated_train_tokens_chars_div_4": stats["train_chars"] // 4,
            "estimated_validation_tokens_chars_div_4": stats["validation_chars"] // 4,
        }
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a Hugging Face training mix for Gemma4 + RWKV-MS token-level experiments."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-cache", default=DEFAULT_HF_CACHE)
    parser.add_argument("--target-train-chars", type=int, default=5_000_000)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--max-record-chars", type=int, default=12_000)
    parser.add_argument("--max-source-errors", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = build_dataset(args)
    print(json.dumps({k: stats[k] for k in ("train_chars", "validation_chars", "train_records", "validation_records")}, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    # Some combinations of datasets/fsspec/hf-xet abort in interpreter finalizers
    # after successful streaming reads. Files are already closed by this point.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
