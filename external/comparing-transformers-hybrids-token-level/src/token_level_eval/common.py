from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency for seed sync only.
    np = None


@dataclass(frozen=True)
class TextRecord:
    doc_id: str
    text: str
    domain: str
    source_path: str | None = None


@dataclass(frozen=True)
class SpanTag:
    start: int
    end: int
    coarse: str
    fine: str
    text: str = ""

    def overlaps(self, start: int, end: int) -> bool:
        return self.start < end and start < self.end

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dtype(name: str):
    import torch

    key = name.lower().strip()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[key]


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_domain(path: Path, fallback: str = "prose") -> str:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in {".html", ".htm", ".xml"}:
        return "html"
    if suffix in {".tex", ".sty", ".bib"}:
        return "latex"
    return fallback


def load_text_records(
    input_path: str | Path,
    *,
    domain: str,
    jsonl_text_key: str = "text",
    jsonl_id_key: str = "id",
    limit: int | None = None,
) -> list[TextRecord]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    records: list[TextRecord] = []

    def add_record(doc_id: str, text: str, record_domain: str, source_path: Path | None = None) -> None:
        if text:
            records.append(
                TextRecord(
                    doc_id=doc_id,
                    text=text,
                    domain=record_domain,
                    source_path=str(source_path) if source_path else None,
                )
            )

    if path.is_dir():
        files = [p for p in sorted(path.rglob("*")) if p.is_file()]
        for file_path in files:
            file_domain = infer_domain(file_path) if domain == "auto" else domain
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            add_record(file_path.stem, text, file_domain, file_path)
            if limit is not None and len(records) >= limit:
                break
        return records

    if path.suffix.lower() == ".jsonl":
        for idx, row in enumerate(read_jsonl(path)):
            text = str(row.get(jsonl_text_key, ""))
            row_id = str(row.get(jsonl_id_key, idx))
            row_domain = str(row.get("domain", domain))
            if row_domain == "auto":
                row_domain = infer_domain(path)
            add_record(row_id, text, row_domain, path)
            if limit is not None and len(records) >= limit:
                break
        return records

    file_domain = infer_domain(path) if domain == "auto" else domain
    text = path.read_text(encoding="utf-8", errors="ignore")
    add_record(path.stem, text, file_domain, path)
    return records[:limit] if limit is not None else records
