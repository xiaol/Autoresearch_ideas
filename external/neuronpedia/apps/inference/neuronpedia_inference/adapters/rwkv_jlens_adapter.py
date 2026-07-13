"""Standalone RWKV adapter for Neuronpedia's JLens stream contract.

The adapter exposes ``/v1/lens/prompt`` with the same NDJSON schema as the main
inference server. Logit Lens reads out captured RWKV block outputs directly;
Jacobian Lens transports them through a fitted RWKV ``J_bar`` artifact first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import unicodedata
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


DEFAULT_MODEL_PATH = "/home/xiaol/X/models/rwkv7-g1/rwkv7a-g1d-0.1b-20260212-ctx8192.pth"
DEFAULT_RWKV_SOURCE = "/home/xiaol/X/rwkv-manifold-steering/src"
RWKV_BOS_TOKEN_ID = 0
RWKV_BOS_TOKEN = "<|endoftext|>"
_RWKV_CHAT_ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}


class LensType(str, Enum):
    LOGIT_LENS = "LOGIT_LENS"
    JACOBIAN_LENS = "JACOBIAN_LENS"


class LensChatMessage(BaseModel):
    role: str
    content: str


class LensSteerToken(BaseModel):
    token: str
    type: LensType


class LensPromptRequest(BaseModel):
    model: str
    type: list[LensType] = Field(default_factory=lambda: [LensType.LOGIT_LENS])
    prompt: str | None = None
    chat: list[LensChatMessage] | None = None
    top_n: int = 8
    layers: list[int] = Field(default_factory=list)
    prepend_bos: bool = False
    enable_thinking: bool = False
    stream: bool = True
    temperature: float = 0.0
    num_completion_tokens: int = 0
    cached_token_ids: list[int] = Field(default_factory=list)
    input_token_ids: list[int] = Field(default_factory=list)
    filter_non_word_tokens: bool = True
    steer_tokens: list[LensSteerToken] = Field(default_factory=list)
    steer_layers: list[int] = Field(default_factory=list)
    steer_strength: float = 0.0
    steer_ablate: bool = False
    swap_token: LensSteerToken | None = None
    steer_generated_tokens: bool = False
    fail_if_busy: bool = False


_MAX_STEER_INJECTION_FRACTION = 1.0
_STEER_NORM_EPS = 1e-12


class RWKVJLensAdapter:
    def __init__(
        self,
        *,
        model_path: str,
        rwkv_source: str,
        device: str,
        compile_cuda: bool,
        jlens_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.rwkv_source = rwkv_source
        self.device = device
        self.compile_cuda = compile_cuda
        self.jlens_path = jlens_path or _default_jlens_path(model_path)
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._functional = None
        self._word_mask = None
        self._valid_token_mask = None
        self._decode_index: dict[str, list[int]] | None = None
        self._jacobians: dict[int, Any] = {}
        self._source_means: dict[int, Any] = {}
        self._target_mean = None
        self._jacobian_device_cache: dict[tuple[int, str], tuple[Any, Any, Any]] = {}
        self._jlens_n_prompts = 0
        self._jlens_status = "not_loaded"
        self._jlens_error: str | None = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if self.rwkv_source not in sys.path:
                sys.path.insert(0, self.rwkv_source)
            import torch
            from torch.nn import functional as F

            from rwkv_manifold_steering.rwkv7a_model import RWKV7AModel
            from rwkv_manifold_steering.tokenizer import RWKVTokenizer

            self._torch = torch
            self._functional = F
            self._tokenizer = RWKVTokenizer()
            self._model = RWKV7AModel(
                self.model_path,
                device=self.device,
                compile_cuda=self.compile_cuda,
            )
            self._load_jacobian_lens()

    def _load_jacobian_lens(self) -> None:
        path = Path(self.jlens_path).expanduser()
        if not path.is_file():
            self._jlens_status = "not_found"
            self._jlens_error = f"RWKV Jacobian lens artifact not found: {path}"
            return

        try:
            checkpoint = self.torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or "J" not in checkpoint:
                raise ValueError("checkpoint has no 'J' matrix dictionary")
            expected_metadata = {
                "format_version": 2,
                "architecture": "rwkv7-g1",
                "activation_site": "block_output",
                "transport": "affine_centered",
                "target_layer": int(self.model.n_layer) - 1,
                "n_layer": int(self.model.n_layer),
                "tokenizer": "rwkv_vocab_v20230424",
                "estimator": "same_position_mean",
            }
            for key, expected in expected_metadata.items():
                actual = checkpoint.get(key)
                if actual != expected:
                    raise ValueError(f"artifact {key}={actual!r}, expected {expected!r}")
            expected_fingerprint = checkpoint.get("model_sha256")
            if not isinstance(expected_fingerprint, str) or len(expected_fingerprint) != 64:
                raise ValueError("artifact has no valid model_sha256 fingerprint")
            actual_fingerprint = _sha256_file(self.model_path)
            if actual_fingerprint != expected_fingerprint:
                raise ValueError(
                    "artifact model_sha256 does not match the served RWKV checkpoint "
                    f"({expected_fingerprint} != {actual_fingerprint})"
                )
            raw_jacobians = checkpoint["J"]
            if not isinstance(raw_jacobians, dict) or not raw_jacobians:
                raise ValueError("checkpoint 'J' must be a non-empty layer-to-matrix dictionary")

            d_model = int(checkpoint.get("d_model", self.model.n_embd))
            if d_model != int(self.model.n_embd):
                raise ValueError(
                    f"artifact d_model={d_model} does not match RWKV d_model={int(self.model.n_embd)}"
                )

            jacobians: dict[int, Any] = {}
            for raw_layer, raw_matrix in raw_jacobians.items():
                layer = int(raw_layer)
                if not (0 <= layer < int(self.model.n_layer) - 1):
                    raise ValueError(
                        f"artifact layer {layer} must be before final RWKV layer {int(self.model.n_layer) - 1}"
                    )
                matrix = raw_matrix.detach().float().cpu()
                if tuple(matrix.shape) != (d_model, d_model):
                    raise ValueError(
                        f"artifact layer {layer} has shape {tuple(matrix.shape)}, expected {(d_model, d_model)}"
                    )
                if not bool(self.torch.isfinite(matrix).all()):
                    raise ValueError(f"artifact layer {layer} contains non-finite values")
                jacobians[layer] = matrix.contiguous()

            metadata_layers = sorted(int(layer) for layer in checkpoint.get("source_layers", []))
            if metadata_layers != sorted(jacobians):
                raise ValueError(
                    f"artifact source_layers={metadata_layers} do not match J keys={sorted(jacobians)}"
                )

            raw_source_means = checkpoint.get("source_means")
            if not isinstance(raw_source_means, dict):
                raise ValueError("checkpoint 'source_means' must be a layer-to-vector dictionary")
            source_means: dict[int, Any] = {}
            for raw_layer, raw_vector in raw_source_means.items():
                layer = int(raw_layer)
                vector = raw_vector.detach().float().cpu().view(-1)
                if tuple(vector.shape) != (d_model,):
                    raise ValueError(
                        f"artifact source mean {layer} has shape {tuple(vector.shape)}, "
                        f"expected {(d_model,)}"
                    )
                if not bool(self.torch.isfinite(vector).all()):
                    raise ValueError(f"artifact source mean {layer} contains non-finite values")
                source_means[layer] = vector.contiguous()
            if sorted(source_means) != sorted(jacobians):
                raise ValueError(
                    f"artifact source mean layers={sorted(source_means)} do not match "
                    f"J keys={sorted(jacobians)}"
                )

            target_mean = checkpoint.get("target_mean")
            if not self.torch.is_tensor(target_mean):
                raise ValueError("checkpoint has no target_mean tensor")
            target_mean = target_mean.detach().float().cpu().view(-1)
            if tuple(target_mean.shape) != (d_model,):
                raise ValueError(
                    f"artifact target_mean has shape {tuple(target_mean.shape)}, expected {(d_model,)}"
                )
            if not bool(self.torch.isfinite(target_mean).all()):
                raise ValueError("artifact target_mean contains non-finite values")

            self._jacobians = dict(sorted(jacobians.items()))
            self._source_means = dict(sorted(source_means.items()))
            self._target_mean = target_mean.contiguous()
            self._jacobian_device_cache = {}
            self._jlens_n_prompts = int(checkpoint.get("n_prompts", 0))
            self._jlens_status = "loaded"
            self._jlens_error = None
        except Exception as exc:
            self._jacobians = {}
            self._source_means = {}
            self._target_mean = None
            self._jacobian_device_cache = {}
            self._jlens_status = "error"
            self._jlens_error = f"Failed to load RWKV Jacobian lens artifact {path}: {exc}"

    @property
    def model(self):
        self.load()
        return self._model

    @property
    def tokenizer(self):
        self.load()
        return self._tokenizer

    @property
    def torch(self):
        self.load()
        return self._torch

    @property
    def F(self):
        self.load()
        return self._functional

    def layers(self, requested: list[int]) -> list[int]:
        model_layers = list(range(int(self.model.n_layer)))
        return _select_layers(model_layers, requested, final_layer=int(self.model.n_layer) - 1)

    def jacobian_layers(self, requested: list[int]) -> list[int]:
        available = [*self._jacobians, int(self.model.n_layer) - 1]
        return _select_layers(available, requested, final_layer=int(self.model.n_layer) - 1)

    def supported_types(self) -> list[LensType]:
        supported = [LensType.LOGIT_LENS]
        if self._jacobians:
            supported.insert(0, LensType.JACOBIAN_LENS)
        return supported

    def encode_request(self, request: LensPromptRequest) -> list[int]:
        if request.input_token_ids:
            return [int(token_id) for token_id in request.input_token_ids]
        if request.chat:
            text = _format_rwkv_chat(request.chat, enable_thinking=request.enable_thinking)
        else:
            text = request.prompt or ""
        return list(self.tokenizer.encode(text))

    def decode_token(self, token_id: int) -> str:
        if int(token_id) == RWKV_BOS_TOKEN_ID:
            return RWKV_BOS_TOKEN
        return self.tokenizer.decode([int(token_id)])

    def token_bytes(self, token_id: int) -> list[int] | None:
        if int(token_id) == RWKV_BOS_TOKEN_ID:
            return None
        raw = self.tokenizer.idx2token.get(int(token_id))
        return list(raw) if isinstance(raw, bytes) else None

    def decode_logits_from_hidden(self, hidden):
        model = self.model
        torch = self.torch
        F = self.F
        z = model.z
        with torch.inference_mode():
            row = hidden.to(device=model.device, dtype=z["ln_out.weight"].dtype).view(-1)
            row = F.layer_norm(row, (model.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
            return row @ z["head.weight"]

    def transport_hidden(self, hidden, layer: int):
        if int(layer) == int(self.model.n_layer) - 1:
            return hidden
        if int(layer) not in self._jacobians:
            raise ValueError(f"RWKV Jacobian lens has no fitted matrix for layer {int(layer)}")
        device = self.model.device
        cache_key = (int(layer), str(device))
        cached = self._jacobian_device_cache.get(cache_key)
        if cached is None:
            matrix = self._jacobians[int(layer)].to(device=device)
            source_mean = self._source_means[int(layer)].to(device=device)
            target_mean = self._target_mean.to(device=device)
            cached = (matrix, source_mean, target_mean)
            self._jacobian_device_cache[cache_key] = cached
        matrix, source_mean, target_mean = cached
        row = hidden.to(device=device, dtype=matrix.dtype).view(-1)
        return (row - source_mean) @ matrix.T + target_mean

    def word_mask(self):
        if self._word_mask is not None:
            return self._word_mask
        torch = self.torch
        flags = torch.zeros(int(self.model.vocab_size), dtype=torch.bool)
        for token_id in self.tokenizer.idx2token:
            try:
                decoded = self.decode_token(token_id)
            except Exception:
                continue
            if 0 <= int(token_id) < int(self.model.vocab_size) and _is_word_like_token(decoded):
                flags[token_id] = True
        self._word_mask = flags
        return flags

    def valid_token_mask(self):
        if self._valid_token_mask is not None:
            return self._valid_token_mask
        torch = self.torch
        flags = torch.zeros(int(self.model.vocab_size), dtype=torch.bool)
        for token_id in self.tokenizer.idx2token:
            if 0 <= int(token_id) < int(self.model.vocab_size):
                flags[int(token_id)] = True
        self._valid_token_mask = flags
        return flags

    def decoded_string_to_ids(self) -> dict[str, list[int]]:
        if self._decode_index is not None:
            return self._decode_index
        index: dict[str, list[int]] = {}
        for token_id in self.tokenizer.idx2token:
            decoded = self.decode_token(int(token_id))
            index.setdefault(decoded, []).append(int(token_id))
        index[RWKV_BOS_TOKEN] = [RWKV_BOS_TOKEN_ID]
        self._decode_index = index
        return index

    def resolve_steer_token_id(self, token: str) -> int:
        index = self.decoded_string_to_ids()
        ids = index.get(token)
        if not ids:
            stripped = token.strip()
            for decoded, candidate_ids in index.items():
                if decoded.strip() == stripped:
                    ids = candidate_ids
                    break
        if not ids:
            raise ValueError(f"Could not resolve RWKV steer token to a vocab id: {token!r}")
        token_id = int(min(ids))
        if not (0 <= token_id < int(self.model.vocab_size)):
            raise ValueError(
                f"RWKV steer token_id {token_id} out of range for vocab size {int(self.model.vocab_size)}"
            )
        return token_id

    def unembed_vector(self, token_id: int):
        # Prepared RWKV head is [d_model, vocab], matching row @ head for logits.
        weight = self.model.z["head.weight"]
        if not (0 <= int(token_id) < int(weight.shape[1])):
            raise ValueError(f"RWKV token_id {token_id} out of range for unembedding size {int(weight.shape[1])}")
        return weight[:, int(token_id)].detach().float()

    def build_steer_deltas(self, steer_tokens: list[LensSteerToken], steer_layers: list[int]) -> dict[int, Any]:
        torch = self.torch
        if not steer_layers:
            steer_layers = self.layers([])
        requested_layers = set(self.layers(steer_layers))
        resolved: list[tuple[LensType, int]] = []
        for spec in steer_tokens:
            if spec.type == LensType.JACOBIAN_LENS and not self._jacobians:
                raise ValueError(self._jlens_error or "RWKV Jacobian lens artifact is not loaded")
            resolved.append((spec.type, self.resolve_steer_token_id(spec.token)))

        deltas: dict[int, Any] = {}
        for layer in requested_layers:
            acc = None
            for lens_type, token_id in resolved:
                direction = self.unembed_vector(token_id)
                if lens_type == LensType.JACOBIAN_LENS and layer != int(self.model.n_layer) - 1:
                    matrix = self._jacobians.get(int(layer))
                    if matrix is None:
                        continue
                    direction = direction @ matrix.to(device=direction.device)
                norm = torch.linalg.vector_norm(direction)
                if norm > 0:
                    direction = direction / norm
                acc = direction if acc is None else acc + direction
            if acc is not None:
                deltas[int(layer)] = acc
        return deltas

    def apply_steer(self, hidden, delta, strength: float, ablate: bool = False):
        torch = self.torch
        d = delta.to(device=hidden.device, dtype=hidden.dtype)
        if ablate:
            norm = torch.linalg.vector_norm(d)
            if norm == 0:
                return hidden
            d_hat = d / norm
            proj = (hidden * d_hat).sum(dim=-1, keepdim=True)
            return hidden - proj * d_hat

        scale = torch.linalg.vector_norm(hidden, dim=-1, keepdim=True)
        injected = (float(strength) * scale) * d
        injected_norm = torch.linalg.vector_norm(injected, dim=-1, keepdim=True)
        max_norm = _MAX_STEER_INJECTION_FRACTION * scale
        clamp_factor = torch.where(
            injected_norm > max_norm,
            max_norm / injected_norm.clamp_min(_STEER_NORM_EPS),
            torch.ones_like(injected_norm),
        )
        return hidden + injected * clamp_factor

    def apply_swap(self, hidden, src_delta, tgt_delta):
        torch = self.torch
        source = src_delta.to(device=hidden.device, dtype=hidden.dtype)
        target = tgt_delta.to(device=hidden.device, dtype=hidden.dtype)
        source_norm = torch.linalg.vector_norm(source)
        target_norm = torch.linalg.vector_norm(target)
        if source_norm == 0 or target_norm == 0:
            return hidden
        source_hat = source / source_norm
        target_hat = target / target_norm
        coeff = (hidden * source_hat).sum(dim=-1, keepdim=True)
        return hidden - coeff * source_hat + coeff * target_hat

    def build_patch_map(
        self,
        *,
        steer_deltas: dict[int, Any],
        steer_strength: float,
        steer_ablate: bool,
        swap_deltas: dict[int, Any],
    ) -> dict[int, Callable[[Any], Any]] | None:
        if swap_deltas and steer_deltas:
            patches: dict[int, Callable[[Any], Any]] = {}
            for layer, target_delta in swap_deltas.items():
                source_delta = steer_deltas.get(layer)
                if source_delta is not None:
                    patches[layer] = (
                        lambda hidden, src=source_delta, tgt=target_delta: self.apply_swap(hidden, src, tgt)
                    )
            return patches or None
        if steer_deltas and (float(steer_strength) != 0.0 or steer_ablate):
            return {
                layer: (
                    lambda hidden, delta=delta: self.apply_steer(
                        hidden,
                        delta,
                        float(steer_strength),
                        steer_ablate,
                    )
                )
                for layer, delta in steer_deltas.items()
            }
        return None

    def top_tokens_and_probs(
        self,
        logits,
        top_n: int,
        filter_non_word_tokens: bool,
    ) -> tuple[list[str], list[float], list[int], list[int]]:
        torch = self.torch
        with torch.inference_mode():
            probs = torch.softmax(logits.float(), dim=-1).detach().cpu()
            valid_mask = self.valid_token_mask()
            scores = probs.clone()
            scores[~valid_mask] = -1
            if filter_non_word_tokens:
                mask = self.word_mask()
                scores[~mask] = -1
                valid_scores = probs.clone()
                valid_scores[~valid_mask] = -1
                true_top = int(valid_scores.argmax())
                if valid_scores[true_top] >= 0:
                    scores[true_top] = probs[true_top]
            k = min(max(int(top_n), 1), int(probs.numel()))
            values = [int(token_id) for token_id in torch.topk(scores, k=k).indices.tolist()]
            valid_scores = probs.clone()
            valid_scores[~valid_mask] = -1
            ranks = [int((valid_scores > probs[token_id]).sum().item()) for token_id in values]
        return (
            [self.decode_token(token_id) for token_id in values],
            [round(float(probs[token_id]), 6) for token_id in values],
            values,
            ranks,
        )

    def build_result_slice(
        self,
        lens_type: LensType,
        hidden_by_layer: list[Any],
        layers: list[int],
        top_n: int,
        filter_non_word_tokens: bool,
    ) -> dict[str, Any]:
        top_tokens: list[list[str]] = []
        top_probs: list[list[float]] = []
        top_token_ids: list[list[int]] = []
        top_ranks: list[list[int]] = []
        for layer in layers:
            hidden = hidden_by_layer[layer]
            if lens_type == LensType.JACOBIAN_LENS:
                hidden = self.transport_hidden(hidden, layer)
            logits = self.decode_logits_from_hidden(hidden)
            tokens, probs, token_ids, ranks = self.top_tokens_and_probs(logits, top_n, filter_non_word_tokens)
            top_tokens.append(tokens)
            top_probs.append(probs)
            top_token_ids.append(token_ids)
            top_ranks.append(ranks)
        return {
            "type": lens_type.value,
            "top_tokens": top_tokens,
            "top_probs": top_probs,
            "top_token_ids": top_token_ids,
            "top_ranks": top_ranks,
        }

    def sample_next(self, logits, temperature: float) -> int:
        torch = self.torch
        with torch.inference_mode():
            row = logits.float().detach()
            valid_mask = self.valid_token_mask().to(device=row.device)
            row = row.masked_fill(~valid_mask, float("-inf"))
            if temperature <= 0:
                return int(row.argmax())
            probs = torch.softmax(row / float(temperature), dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())

    def run_messages(self, request: LensPromptRequest) -> Iterator[dict[str, Any]]:
        self.load()
        requested_types = list(dict.fromkeys(request.type))
        available_types = set(self.supported_types())
        unavailable_types = [lens_type for lens_type in requested_types if lens_type not in available_types]
        if unavailable_types:
            labels = ", ".join(lens_type.value for lens_type in unavailable_types)
            yield {
                "kind": "error",
                "error": (
                    f"Requested RWKV lens types are unavailable: {labels}. "
                    + (self._jlens_error or "RWKV Jacobian lens artifact is not loaded")
                ),
            }
            return
        supported_types = requested_types

        token_ids = self.encode_request(request)
        prompt_len = len(token_ids)
        max_generation = 0 if request.input_token_ids else max(0, int(request.num_completion_tokens))
        layers_by_type = {
            lens_type: (
                self.jacobian_layers(request.layers)
                if lens_type == LensType.JACOBIAN_LENS
                else self.layers(request.layers)
            )
            for lens_type in supported_types
        }
        reuse_len = _common_prefix_len(token_ids, request.cached_token_ids)
        swap_active = request.swap_token is not None and len(request.steer_tokens) > 0
        steer_active = len(request.steer_tokens) > 0 and (request.steer_strength != 0.0 or request.steer_ablate)
        steer_deltas: dict[int, Any] = {}
        swap_deltas: dict[int, Any] = {}
        if steer_active or swap_active:
            try:
                steer_deltas = self.build_steer_deltas(request.steer_tokens, request.steer_layers)
                if swap_active and request.swap_token is not None:
                    swap_deltas = self.build_steer_deltas([request.swap_token], request.steer_layers)
            except Exception as exc:
                yield {"kind": "error", "error": str(exc)}
                return
            if steer_deltas or swap_deltas:
                reuse_len = 0

        prompt_patch = self.build_patch_map(
            steer_deltas=steer_deltas,
            steer_strength=request.steer_strength,
            steer_ablate=request.steer_ablate,
            swap_deltas=swap_deltas,
        )
        generation_patch = prompt_patch if request.steer_generated_tokens else None

        yield {
            "kind": "meta",
            "model": request.model,
            "types": [lens_type.value for lens_type in supported_types],
            "layers_by_type": {
                lens_type.value: layers_by_type[lens_type] for lens_type in supported_types
            },
            "top_n": request.top_n,
            "prompt_len": prompt_len,
            "num_completion_tokens": max_generation,
            "temperature": request.temperature,
            # G1's checkpoint-specific prompt guide and reference inference
            # scripts encode the template directly, without token 0.
            "prepend_bos": False,
            "reuse_len": reuse_len,
        }
        yield {
            "kind": "prompt",
            "tokens": [
                _token_payload(
                    position=position,
                    token=self.decode_token(token_id),
                    token_id=int(token_id),
                    is_generated=False,
                    token_bytes=self.token_bytes(token_id),
                )
                for position, token_id in enumerate(token_ids)
            ],
        }

        state = None
        last_logits = None
        completion: list[str] = []

        for position, token_id in enumerate(token_ids):
            out = self.model.forward(int(token_id), state, collect_layers=True, patch=prompt_patch)
            state = out.state
            last_logits = out.logits
            if position < reuse_len:
                continue
            yield self._token_message(
                position,
                int(token_id),
                False,
                out.hidden_by_layer or [],
                supported_types,
                layers_by_type,
                request,
            )

        for _ in range(max_generation):
            if last_logits is None:
                break
            next_id = self.sample_next(last_logits, request.temperature)
            piece = self.decode_token(next_id)
            if _is_rwkv_generation_stop(next_id, piece, completion):
                break
            out = self.model.forward(next_id, state, collect_layers=True, patch=generation_patch)
            state = out.state
            last_logits = out.logits
            position = len(token_ids)
            token_ids.append(next_id)
            completion.append(piece)
            yield self._token_message(
                position,
                next_id,
                True,
                out.hidden_by_layer or [],
                supported_types,
                layers_by_type,
                request,
            )

        yield {
            "kind": "done",
            "seq_len": len(token_ids),
            "prompt_len": prompt_len,
            "vocab_size": int(self.model.vocab_size),
            "completion": "".join(completion),
        }

    def _token_message(
        self,
        position: int,
        token_id: int,
        is_generated: bool,
        hidden_by_layer: list[Any],
        supported_types: list[LensType],
        layers_by_type: dict[LensType, list[int]],
        request: LensPromptRequest,
    ) -> dict[str, Any]:
        return _token_payload(
            position=position,
            token=self.decode_token(token_id),
            token_id=int(token_id),
            is_generated=is_generated,
            token_bytes=self.token_bytes(token_id),
            results=[
                self.build_result_slice(
                    lens_type,
                    hidden_by_layer,
                    layers_by_type[lens_type],
                    request.top_n,
                    request.filter_non_word_tokens,
                )
                for lens_type in supported_types
            ],
        )


def _token_payload(
    *,
    position: int,
    token: str,
    token_id: int,
    is_generated: bool,
    token_bytes: list[int] | None,
    results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "position": position,
        "token": token,
        "id": token_id,
        "is_generated": is_generated,
    }
    if token_bytes is not None:
        payload["token_bytes"] = token_bytes
    if results is not None:
        payload["kind"] = "token"
        payload["results"] = results
    return payload


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        count += 1
    return count


def _default_jlens_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    return str(path.with_name(f"{path.stem}_jacobian_lens.pt"))


def _sha256_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _select_layers(available: list[int], requested: list[int], *, final_layer: int) -> list[int]:
    available_set = {int(layer) for layer in available}
    if requested:
        selected = available_set.intersection(int(layer) for layer in requested)
    else:
        selected = set(available_set)
    if final_layer in available_set:
        selected.add(final_layer)
    return sorted(selected)


def _clean_rwkv_chat_text(text: str, role: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if role in ("system", "user"):
        # G1 uses blank lines as turn separators. Official templates therefore
        # collapse them only in externally supplied System/User text. Assistant
        # history must retain its paragraphs or the next-turn token prefix drifts.
        return re.sub(r"\n{2,}", "\n", normalized)
    return normalized


def _format_rwkv_chat(chat: list[LensChatMessage], *, enable_thinking: bool = False) -> str:
    # Mirrors the checkpoint-specific RWKV7-G1x prompt guide. System messages are
    # collected into one leading prompt, completed turns are separated by blank
    # lines, and the final '>' is deliberately left for the model to generate.
    parts: list[str] = []
    system_prompt = ""
    for message in chat:
        if message.role.lower() == "system":
            system_prompt = _clean_rwkv_chat_text(message.content, "system")
    if system_prompt:
        parts.append(f"System: {system_prompt}")

    for message in chat:
        role = message.role.lower()
        if role == "system":
            continue
        label = _RWKV_CHAT_ROLE_LABELS.get(role)
        if label is None:
            raise ValueError(f"Unsupported RWKV chat role: {message.role!r}")
        content = _clean_rwkv_chat_text(message.content, role)
        parts.append(f"{label}: {content}" if content else f"{label}:")

    if not chat or chat[-1].role.lower() != "assistant":
        parts.append("Assistant: <think" if enable_thinking else "Assistant: <think>\n</think")

    return "\n\n".join(parts).rstrip()


def _is_rwkv_generation_stop(token_id: int, piece: str, completion: list[str]) -> bool:
    if int(token_id) in (RWKV_BOS_TOKEN_ID, 261):
        return True
    if "\n\n" in piece and piece.strip() == "":
        return True
    return bool(completion and completion[-1].endswith("\n") and piece.startswith("\n"))


def _is_word_like_token(token: str) -> bool:
    stripped = token.strip()
    if stripped == "":
        return False
    if "<|" in stripped or (stripped.startswith("<") and stripped.endswith(">")):
        return False
    chars = list(stripped)
    for index, char in enumerate(chars):
        if unicodedata.category(char)[0] in ("L", "N"):
            continue
        if 0 < index < len(chars) - 1 and char in ("'", "-"):
            continue
        return False
    return True


def _ndjson(messages: Iterator[dict[str, Any]], release_lock: threading.Lock | None = None) -> Iterator[bytes]:
    try:
        for message in messages:
            yield (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
    finally:
        if release_lock is not None:
            release_lock.release()


def create_app(adapter: RWKVJLensAdapter) -> FastAPI:
    app = FastAPI(title="Neuronpedia RWKV JLens Adapter")
    request_lock = threading.Lock()

    @app.get("/health")
    def health():
        supported = [LensType.LOGIT_LENS.value]
        if adapter._jacobians:
            supported.insert(0, LensType.JACOBIAN_LENS.value)
        return {
            "status": "healthy",
            "loaded": adapter._model is not None,
            "model_path": adapter.model_path,
            "supports": supported,
            "jlens_path": adapter.jlens_path,
            "jlens_status": adapter._jlens_status,
            "jlens_n_prompts": adapter._jlens_n_prompts,
            "jlens_error": adapter._jlens_error,
        }

    @app.post("/v1/lens/prompt")
    def lens_prompt(request: LensPromptRequest):
        acquired = request_lock.acquire(blocking=not request.fail_if_busy)
        if not acquired:
            return JSONResponse({"error": "RWKV adapter is busy"}, status_code=429)

        if request.stream:
            return StreamingResponse(
                _ndjson(adapter.run_messages(request), release_lock=request_lock),
                media_type="application/x-ndjson",
            )

        try:
            meta = None
            tokens = []
            done = None
            for message in adapter.run_messages(request):
                if message["kind"] == "error":
                    return JSONResponse({"error": message["error"]}, status_code=500)
                if message["kind"] == "meta":
                    meta = message
                elif message["kind"] == "token":
                    tokens.append(message)
                elif message["kind"] == "done":
                    done = message
            if meta is None or done is None:
                return JSONResponse({"error": "RWKV adapter produced incomplete data"}, status_code=500)
            return {"meta": meta, "tokens": tokens, "done": done}
        finally:
            request_lock.release()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local RWKV JLens adapter.")
    parser.add_argument("--host", default=os.environ.get("RWKV_JLENS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RWKV_JLENS_PORT", "5003")))
    parser.add_argument("--model-path", default=os.environ.get("RWKV_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--rwkv-source", default=os.environ.get("RWKV_SOURCE", DEFAULT_RWKV_SOURCE))
    parser.add_argument("--jlens-path", default=os.environ.get("RWKV_JLENS_PATH"))
    parser.add_argument("--device", default=os.environ.get("RWKV_DEVICE", "cuda"))
    parser.add_argument("--no-compile-cuda", action="store_true")
    parser.add_argument("--eager-load", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = RWKVJLensAdapter(
        model_path=str(Path(args.model_path).expanduser()),
        rwkv_source=str(Path(args.rwkv_source).expanduser()),
        device=args.device,
        compile_cuda=not args.no_compile_cuda,
        jlens_path=(str(Path(args.jlens_path).expanduser()) if args.jlens_path else None),
    )
    if args.eager_load:
        adapter.load()
    app = create_app(adapter)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
