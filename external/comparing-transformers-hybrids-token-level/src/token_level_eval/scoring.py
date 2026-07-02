from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from token_level_eval.common import TextRecord, default_device, parse_dtype
from token_level_eval.tagging import (
    approximate_offsets,
    copy_features,
    sorted_tag_names,
    tag_source,
    tags_for_span,
    word_position,
)


@dataclass(frozen=True)
class ModelLoadConfig:
    dtype: str = "bfloat16"
    device: str = "auto"
    device_map: str | None = None
    trust_remote_code: bool = True
    local_files_only: bool = False
    attn_implementation: str | None = None


def load_tokenizer(tokenizer_name_or_path: str, *, trust_remote_code: bool = True, local_files_only: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_name_or_path: str, cfg: ModelLoadConfig):
    kwargs: dict[str, Any] = {
        "torch_dtype": parse_dtype(cfg.dtype),
        "trust_remote_code": cfg.trust_remote_code,
        "local_files_only": cfg.local_files_only,
    }
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    if cfg.device_map:
        kwargs["device_map"] = cfg.device_map
        kwargs["low_cpu_mem_usage"] = True
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    else:
        device = default_device() if cfg.device == "auto" else cfg.device
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs).to(device)
    model.eval()
    return model


def infer_input_device(model: Any) -> torch.device:
    if hasattr(model, "hf_device_map"):
        device_map = getattr(model, "hf_device_map")
        for device_name in device_map.values():
            if isinstance(device_name, str) and device_name not in {"cpu", "disk"}:
                return torch.device(device_name)
    return next(model.parameters()).device


def encode_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]], list[str]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError):
        encoded = tokenizer(
            text,
            add_special_tokens=False,
        )
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded.get("offset_mapping", [])]
    token_texts = tokenizer.convert_ids_to_tokens(input_ids)
    if len(offsets) != len(input_ids) or all(start == end for start, end in offsets):
        offsets = approximate_offsets(text, token_texts)
    return input_ids, offsets, token_texts


@torch.inference_mode()
def score_token_nlls(
    model: Any,
    token_ids: list[int],
    *,
    seq_len: int,
    progress_label: str | None = None,
    before_window: Callable[[], None] | None = None,
) -> dict[int, float]:
    if len(token_ids) < 2:
        return {}
    input_device = infer_input_device(model)
    losses: dict[int, float] = {}
    starts = range(0, len(token_ids) - 1, max(seq_len, 2))
    iterator = tqdm(list(starts), desc=progress_label, leave=False) if progress_label else starts
    for start in iterator:
        if before_window is not None:
            before_window()
        end = min(start + seq_len, len(token_ids))
        window = token_ids[start:end]
        if len(window) < 2:
            continue
        input_ids = torch.tensor(window, dtype=torch.long, device=input_device).unsqueeze(0)
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[:, :-1, :].float()
        labels = input_ids[:, 1:]
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )
        for offset, loss in enumerate(token_losses.detach().cpu().tolist(), start=1):
            losses[start + offset] = float(loss)
    return losses


def unload_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def paired_token_rows(
    record: TextRecord,
    *,
    tokenizer: Any,
    transformer_losses: dict[int, float],
    hybrid_losses: dict[int, float],
    token_ids: list[int],
    offsets: list[tuple[int, int]],
    token_texts: list[str],
    max_copy_ngram: int,
) -> Iterator[dict[str, Any]]:
    source_tags = tag_source(record.text, record.domain)
    copy_rows, prev_distances, token_counts = copy_features(token_ids, max_copy_ngram)
    total = max(len(token_ids) - 1, 1)

    for pos in sorted(set(transformer_losses) & set(hybrid_losses)):
        if pos >= len(token_ids):
            continue
        start, end = offsets[pos] if pos < len(offsets) else (0, 0)
        token_text = record.text[start:end] if end > start else token_texts[pos]
        matched_tags = tags_for_span(source_tags, start, end)
        tags, fine_tags, aggregate_tags = sorted_tag_names(matched_tags)
        if not tags:
            tags = ["Untagged"]
            fine_tags = ["Untagged/Untagged"]
            aggregate_tags = ["Other"]
        loss_transformer = transformer_losses[pos]
        loss_hybrid = hybrid_losses[pos]
        mean_loss = 0.5 * (loss_transformer + loss_hybrid)
        row: dict[str, Any] = {
            "doc_id": record.doc_id,
            "domain": record.domain,
            "source_path": record.source_path,
            "position": pos,
            "rel_pos": pos / total,
            "token_id": int(token_ids[pos]),
            "token_text": token_text,
            "token_start": int(start),
            "token_end": int(end),
            "tags": tags,
            "fine_tags": fine_tags,
            "aggregate_tags": aggregate_tags,
            "primary_tag": tags[0],
            "primary_fine_tag": fine_tags[0],
            "primary_aggregate_tag": aggregate_tags[0],
            "word_position": word_position(source_tags, start, end),
            "loss_transformer": loss_transformer,
            "loss_hybrid": loss_hybrid,
            "loss_gap": loss_transformer - loss_hybrid,
            "mean_loss": mean_loss,
            "prev_distance": prev_distances[pos],
            "token_frequency_doc": int(token_counts[token_ids[pos]]),
        }
        row.update(copy_rows[pos])
        yield row


def score_records(
    records: list[TextRecord],
    *,
    transformer_model: str,
    hybrid_model: str,
    tokenizer_name_or_path: str | None,
    model_cfg: ModelLoadConfig,
    seq_len: int,
    max_copy_ngram: int,
) -> Iterator[dict[str, Any]]:
    tokenizer = load_tokenizer(
        tokenizer_name_or_path or transformer_model,
        trust_remote_code=model_cfg.trust_remote_code,
        local_files_only=model_cfg.local_files_only,
    )
    encoded_records = []
    for record in records:
        token_ids, offsets, token_texts = encode_with_offsets(tokenizer, record.text)
        encoded_records.append((record, token_ids, offsets, token_texts))

    transformer = load_causal_lm(transformer_model, model_cfg)
    transformer_by_doc: dict[str, dict[int, float]] = {}
    for record, token_ids, _, _ in encoded_records:
        transformer_by_doc[record.doc_id] = score_token_nlls(
            transformer,
            token_ids,
            seq_len=seq_len,
            progress_label=f"transformer:{record.doc_id}",
        )
    unload_model(transformer)

    hybrid = load_causal_lm(hybrid_model, model_cfg)
    hybrid_by_doc: dict[str, dict[int, float]] = {}
    for record, token_ids, _, _ in encoded_records:
        hybrid_by_doc[record.doc_id] = score_token_nlls(
            hybrid,
            token_ids,
            seq_len=seq_len,
            progress_label=f"hybrid:{record.doc_id}",
        )
    unload_model(hybrid)

    for record, token_ids, offsets, token_texts in encoded_records:
        yield from paired_token_rows(
            record,
            tokenizer=tokenizer,
            transformer_losses=transformer_by_doc[record.doc_id],
            hybrid_losses=hybrid_by_doc[record.doc_id],
            token_ids=token_ids,
            offsets=offsets,
            token_texts=token_texts,
            max_copy_ngram=max_copy_ngram,
        )
