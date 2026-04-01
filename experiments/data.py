from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch

BYTE_PAD = 256
BYTE_BOS = 257
BYTE_EOS = 258
BYTE_VOCAB_SIZE = 259

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class ByteCorpus:
    name: str
    train_tokens: torch.Tensor
    valid_tokens: torch.Tensor
    test_tokens: torch.Tensor
    vocab_size: int = BYTE_VOCAB_SIZE
    pad_token_id: int = BYTE_PAD


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _encode_text(text: str) -> List[int]:
    payload = text.encode("utf-8", errors="ignore")
    return [BYTE_BOS, *payload, BYTE_EOS]


def _join_records(records: Iterable[str]) -> str:
    kept = [record.strip() for record in records if record and record.strip()]
    return "\n".join(kept)


def _save_metadata(path: Path, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_wikitext2(cache_dir: Path) -> ByteCorpus:
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=str(cache_dir / "hf"))
    train_text = _join_records(dataset["train"]["text"])
    valid_text = _join_records(dataset["validation"]["text"])
    test_text = _join_records(dataset["test"]["text"])
    return ByteCorpus(
        name="wikitext2-bytes",
        train_tokens=torch.tensor(_encode_text(train_text), dtype=torch.long),
        valid_tokens=torch.tensor(_encode_text(valid_text), dtype=torch.long),
        test_tokens=torch.tensor(_encode_text(test_text), dtype=torch.long),
    )


def _load_tinyshakespeare(cache_dir: Path) -> ByteCorpus:
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_path = cache_dir / "tinyshakespeare.txt"
    if not text_path.exists():
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, text_path)

    text = text_path.read_text(encoding="utf-8")
    train_end = int(0.9 * len(text))
    valid_end = int(0.95 * len(text))
    return ByteCorpus(
        name="tinyshakespeare-bytes",
        train_tokens=torch.tensor(_encode_text(text[:train_end]), dtype=torch.long),
        valid_tokens=torch.tensor(_encode_text(text[train_end:valid_end]), dtype=torch.long),
        test_tokens=torch.tensor(_encode_text(text[valid_end:]), dtype=torch.long),
    )


def load_byte_corpus(name: str = "wikitext2", cache_dir: str | Path = "results/cache") -> ByteCorpus:
    cache_root = Path(cache_dir)
    try:
        if name == "wikitext2":
            corpus = _load_wikitext2(cache_root)
        elif name == "tinyshakespeare":
            corpus = _load_tinyshakespeare(cache_root)
        else:
            raise ValueError(f"Unknown corpus: {name}")
    except Exception:
        corpus = _load_tinyshakespeare(cache_root)

    metadata = {
        "name": corpus.name,
        "train_tokens": int(corpus.train_tokens.numel()),
        "valid_tokens": int(corpus.valid_tokens.numel()),
        "test_tokens": int(corpus.test_tokens.numel()),
        "vocab_size": corpus.vocab_size,
    }
    _save_metadata(cache_root / f"{corpus.name}_metadata.json", metadata)
    return corpus


def random_lm_batch(
    tokens: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = tokens.numel() - seq_len - 1
    if max_start <= 0:
        raise ValueError("Corpus is too short for the requested sequence length.")

    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    sequences = [tokens[start : start + seq_len + 1] for start in starts.tolist()]
    batch = torch.stack(sequences, dim=0)
    x = batch[:, :-1].to(device)
    y = batch[:, 1:].to(device)
    return x, y


def iter_eval_batches(
    tokens: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
):
    usable_tokens = ((tokens.numel() - 1) // seq_len) * seq_len
    inputs = tokens[:usable_tokens]
    targets = tokens[1 : usable_tokens + 1]
    x_chunks = inputs.view(-1, seq_len)
    y_chunks = targets.view(-1, seq_len)

    total_chunks = x_chunks.size(0)
    batches_emitted = 0
    for start in range(0, total_chunks, batch_size):
        if max_batches is not None and batches_emitted >= max_batches:
            break
        end = min(start + batch_size, total_chunks)
        yield x_chunks[start:end].to(device), y_chunks[start:end].to(device)
        batches_emitted += 1
