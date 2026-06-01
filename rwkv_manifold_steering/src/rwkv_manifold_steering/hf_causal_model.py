from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-0.8B-Base"


@dataclass(frozen=True)
class HiddenPatch:
    layer: int
    hidden: torch.Tensor


@dataclass
class TransformerForwardOutput:
    logits: torch.Tensor
    hidden_by_layer: list[torch.Tensor] | None = None


class HFCausalTransformer:
    """Hugging Face causal LM wrapper with last-token block-output patching."""

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_QWEN_MODEL,
        *,
        device: str | torch.device = "cuda",
        dtype: str = "auto",
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        model_dtype = self._resolve_dtype(dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype=model_dtype,
        ).to(self.device)
        self.model.eval()
        self.layers = self._resolve_layers()
        self.n_layer = len(self.layers)
        self.n_embd = int(self.model.config.hidden_size)

    def _resolve_dtype(self, dtype: str) -> torch.dtype | str:
        if dtype == "auto":
            if self.device.type == "cuda":
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        if dtype in {"float16", "fp16"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"unsupported dtype: {dtype}")

    def _resolve_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise ValueError("could not find decoder block list on HF model")

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def encode_label(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    @torch.inference_mode()
    def forward(
        self,
        tokens: int | Iterable[int] | torch.Tensor,
        *,
        collect_layers: bool = False,
        patch: HiddenPatch | None = None,
    ) -> TransformerForwardOutput:
        ids = self._normalize_tokens(tokens)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        handle = None
        if patch is not None:
            handle = self.layers[patch.layer].register_forward_hook(
                self._make_patch_hook(patch)
            )
        try:
            outputs = self.model(
                input_ids,
                use_cache=False,
                output_hidden_states=collect_layers,
                return_dict=True,
            )
        finally:
            if handle is not None:
                handle.remove()
        hidden_by_layer = None
        if collect_layers:
            hidden_by_layer = [
                hidden[0, -1, :].detach().float().cpu()
                for hidden in outputs.hidden_states[1:]
            ]
        return TransformerForwardOutput(
            logits=outputs.logits[0, -1, :],
            hidden_by_layer=hidden_by_layer,
        )

    def _normalize_tokens(self, tokens: int | Iterable[int] | torch.Tensor) -> list[int]:
        if isinstance(tokens, int):
            return [tokens]
        if isinstance(tokens, torch.Tensor):
            return [int(x) for x in tokens.detach().cpu().view(-1).tolist()]
        return [int(x) for x in tokens]

    def _make_patch_hook(self, patch: HiddenPatch):
        def _hook(_module, _inputs, output):
            replacement = patch.hidden.to(device=self.device).view(-1)
            if replacement.numel() != self.n_embd:
                raise ValueError(
                    f"patch for layer {patch.layer} has {replacement.numel()} values, "
                    f"expected {self.n_embd}"
                )
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[0, -1, :] = replacement.to(dtype=hidden.dtype)
                return (hidden,) + output[1:]
            hidden = output.clone()
            hidden[0, -1, :] = replacement.to(dtype=hidden.dtype)
            return hidden

        return _hook
