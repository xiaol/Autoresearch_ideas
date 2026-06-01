from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load


DEFAULT_MODEL_PATH = os.environ.get("RWKV_MODEL_PATH", "models/rwkv7-0.1b.pth")
DEFAULT_HEAD_SIZE = 64
DTYPE = torch.float16

_WKV7S_LOADED = False


@dataclass(frozen=True)
class HiddenPatch:
    """Patch the last-token block output at ``layer`` with ``hidden``."""

    layer: int
    hidden: torch.Tensor


@dataclass
class RWKVForwardOutput:
    logits: torch.Tensor
    state: list[torch.Tensor]
    hidden_by_layer: list[torch.Tensor] | None = None


def _model_path_with_suffix(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser()
    if path.suffix != ".pth":
        path = path.with_suffix(".pth")
    return path


def _load_wkv7s(head_size: int = DEFAULT_HEAD_SIZE, verbose: bool = False) -> None:
    global _WKV7S_LOADED
    if _WKV7S_LOADED:
        return
    cuda_dir = resources.files(__package__).joinpath("cuda")
    load(
        name=f"rwkv_manifold_wkv7s_n{head_size}",
        sources=[
            str(cuda_dir.joinpath("wkv7s_op.cpp")),
            str(cuda_dir.joinpath("wkv7s.cu")),
        ],
        is_python_module=False,
        verbose=verbose,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas",
            "-O3",
            "--extra-device-vectorization",
            f"-D_N_={head_size}",
        ],
    )
    _WKV7S_LOADED = True


class WKV7S(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):  # type: ignore[override]
        with torch.no_grad():
            time_steps, channels = r.size()
            heads = channels // DEFAULT_HEAD_SIZE
            if DEFAULT_HEAD_SIZE != channels // heads:
                raise ValueError("bad RWKV head size")
            tensors = [r, w, k, v, a, b]
            if any(x.dtype != DTYPE for x in tensors):
                raise TypeError("RWKV CUDA kernel expects float16 activations")
            if any(not x.is_contiguous() for x in tensors):
                raise ValueError("RWKV CUDA kernel expects contiguous tensors")
            y = torch.empty(
                (time_steps, channels),
                device=k.device,
                dtype=DTYPE,
                requires_grad=False,
                memory_format=torch.contiguous_format,
            )
            torch.ops.wkv7s.forward(1, time_steps, channels, heads, state, r, w, k, v, a, b, y)
            return y


def rwkv7_op(state, r, w, k, v, a, b):
    return WKV7S.apply(state, r, w, k, v, a, b)


class RWKV7AModel(nn.Module):
    """Minimal RWKV-7 G1 inference wrapper with hidden-state capture and patching.

    The captured/patchable activation is the hidden vector ``x`` after each full
    block's time-mix and channel-mix residual updates. This is the closest RWKV
    analogue of a Transformer block-output residual stream.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        device: str | torch.device = "cuda",
        compile_cuda: bool = True,
        verbose_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.model_path = _model_path_with_suffix(model_path)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("This RWKV-7 wrapper currently requires CUDA.")
        if compile_cuda:
            _load_wkv7s(DEFAULT_HEAD_SIZE, verbose=verbose_kernel)

        raw = torch.load(self.model_path, map_location=self.device)
        self.n_head, self.head_size = raw["blocks.0.att.r_k"].shape
        if self.head_size != DEFAULT_HEAD_SIZE:
            raise ValueError(f"expected head size {DEFAULT_HEAD_SIZE}, got {self.head_size}")
        self.z = self._prepare_weights(raw)
        self.n_embd = int(self.z["emb.weight"].shape[1])
        self.vocab_size = int(self.z["emb.weight"].shape[0])
        self.n_layer = 1 + max(
            int(key.split(".")[1]) for key in self.z if key.startswith("blocks.")
        )
        self.eval()

    def _prepare_weights(self, z: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z = dict(z)
        n_embd = int(z["emb.weight"].shape[1])
        n_layer = 1 + max(int(key.split(".")[1]) for key in z if key.startswith("blocks."))

        for key in list(z.keys()):
            if (
                "key.weight" in key
                or "value.weight" in key
                or "receptance.weight" in key
                or "output.weight" in key
                or "head.weight" in key
            ):
                z[key] = z[key].t()
            z[key] = z[key].squeeze().to(device=self.device, dtype=DTYPE).contiguous()
            if key.endswith("att.r_k"):
                z[key] = z[key].flatten().contiguous()

        z["emb.weight"] = F.layer_norm(
            z["emb.weight"],
            (n_embd,),
            weight=z["blocks.0.ln0.weight"],
            bias=z["blocks.0.ln0.bias"],
        ).contiguous()

        for layer in range(n_layer):
            z[f"blocks.{layer}.ffn.s_emb.weight"] = (
                z[f"blocks.{layer}.ffn.s_emb.weight"]
                + z["emb.weight"] @ z[f"blocks.{layer}.ffn.s_emb_x.weight"].t()
            ).contiguous()

        z["blocks.0.att.v0"] = z["blocks.0.att.a0"]
        z["blocks.0.att.v1"] = z["blocks.0.att.a1"]
        z["blocks.0.att.v2"] = z["blocks.0.att.a2"]
        return z

    def initial_state(self) -> list[torch.Tensor]:
        return [
            tensor
            for layer in range(self.n_layer)
            for tensor in (
                torch.zeros(
                    self.n_embd,
                    dtype=DTYPE,
                    requires_grad=False,
                    device=self.device,
                ),
                torch.zeros(
                    (self.n_embd // self.head_size, self.head_size, self.head_size),
                    dtype=torch.float32,
                    requires_grad=False,
                    device=self.device,
                ),
                torch.zeros(
                    self.n_embd,
                    dtype=DTYPE,
                    requires_grad=False,
                    device=self.device,
                ),
            )
        ]

    @torch.inference_mode()
    def forward(
        self,
        tokens: int | Iterable[int] | torch.Tensor,
        state: list[torch.Tensor] | None = None,
        *,
        collect_layers: bool = False,
        patch: HiddenPatch | dict[int, torch.Tensor] | None = None,
        full_output: bool = False,
    ) -> RWKVForwardOutput:
        if state is None:
            state = self.initial_state()
        idx = self._normalize_tokens(tokens)
        logits, state, hidden_by_layer = self._forward_seq(
            idx,
            state,
            collect_layers=collect_layers,
            patch=patch,
            full_output=full_output,
        )
        return RWKVForwardOutput(logits=logits, state=state, hidden_by_layer=hidden_by_layer)

    def _normalize_tokens(self, tokens: int | Iterable[int] | torch.Tensor) -> list[int]:
        if isinstance(tokens, int):
            return [tokens]
        if isinstance(tokens, torch.Tensor):
            return [int(x) for x in tokens.detach().cpu().view(-1).tolist()]
        return [int(x) for x in tokens]

    def _patches(
        self, patch: HiddenPatch | dict[int, torch.Tensor] | None
    ) -> dict[int, torch.Tensor]:
        if patch is None:
            return {}
        if isinstance(patch, HiddenPatch):
            return {patch.layer: patch.hidden}
        return patch

    def _forward_seq(
        self,
        idx: list[int],
        state: list[torch.Tensor],
        *,
        collect_layers: bool,
        patch: HiddenPatch | dict[int, torch.Tensor] | None,
        full_output: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor] | None]:
        if not idx:
            raise ValueError("tokens cannot be empty")

        z = self.z
        x = z["emb.weight"][idx].contiguous()
        v_first = torch.empty_like(x)
        patches = self._patches(patch)
        hidden_by_layer: list[torch.Tensor] | None = [] if collect_layers else None

        for layer in range(self.n_layer):
            block = f"blocks.{layer}."
            att = f"{block}att."
            ffn = f"{block}ffn."

            xx = F.layer_norm(
                x,
                (self.n_embd,),
                weight=z[f"{block}ln1.weight"],
                bias=z[f"{block}ln1.bias"],
            ).contiguous()

            xx, state[layer * 3 + 0], state[layer * 3 + 1], v_first = time_mix_seq(
                layer,
                self.n_head,
                self.head_size,
                xx,
                state[layer * 3 + 0],
                v_first,
                state[layer * 3 + 1],
                z[f"{att}x_r"],
                z[f"{att}x_w"],
                z[f"{att}x_k"],
                z[f"{att}x_v"],
                z[f"{att}x_a"],
                z[f"{att}x_g"],
                z[f"{att}w0"],
                z[f"{att}w1"],
                z[f"{att}w2"],
                z[f"{att}a0"],
                z[f"{att}a1"],
                z[f"{att}a2"],
                z[f"{att}v0"],
                z[f"{att}v1"],
                z[f"{att}v2"],
                z[f"{att}g1"],
                z[f"{att}g2"],
                z[f"{att}k_k"],
                z[f"{att}k_a"],
                z[f"{att}r_k"],
                z[f"{att}receptance.weight"],
                z[f"{att}key.weight"],
                z[f"{att}value.weight"],
                z[f"{att}output.weight"],
                z[f"{att}ln_x.weight"],
                z[f"{att}ln_x.bias"],
            )
            x = x + xx

            xx = F.layer_norm(
                x,
                (self.n_embd,),
                weight=z[f"{block}ln2.weight"],
                bias=z[f"{block}ln2.bias"],
            ).contiguous()

            xx, state[layer * 3 + 2] = channel_mix_seq(
                xx,
                state[layer * 3 + 2],
                z[f"{ffn}x_k"],
                z[f"{ffn}key.weight"],
                z[f"{ffn}value.weight"],
                z[f"{ffn}s_emb.weight"][idx],
                z[f"{ffn}s1"],
                z[f"{ffn}s2"],
                z[f"{ffn}s0"],
            )
            x = x + xx

            if layer in patches:
                replacement = patches[layer].to(device=x.device, dtype=x.dtype).view(-1)
                if replacement.numel() != self.n_embd:
                    raise ValueError(
                        f"patch for layer {layer} has {replacement.numel()} values, "
                        f"expected {self.n_embd}"
                    )
                x = x.clone()
                x[-1, :] = replacement

            if hidden_by_layer is not None:
                hidden_by_layer.append(x[-1, :].detach().float().cpu())

        if not full_output:
            x = x[-1, :]
        x = F.layer_norm(x, (self.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
        logits = x @ z["head.weight"]
        return logits, state, hidden_by_layer


def time_mix_seq(
    layer_id: int,
    heads: int,
    head_size: int,
    x,
    x_prev,
    v_first,
    state,
    x_r,
    x_w,
    x_k,
    x_v,
    x_a,
    x_g,
    w0,
    w1,
    w2,
    a0,
    a1,
    a2,
    v0,
    v1,
    v2,
    g1,
    g2,
    k_k,
    k_a,
    r_k,
    receptance_weight,
    key_weight,
    value_weight,
    output_weight,
    ln_weight,
    ln_bias,
):
    time_steps = x.shape[0]
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
    xr, xw, xk, xv, xa, xg = (
        x + xx * x_r,
        x + xx * x_w,
        x + xx * x_k,
        x + xx * x_v,
        x + xx * x_a,
        x + xx * x_g,
    )

    r = (xr @ receptance_weight).contiguous()
    w = (torch.tanh(xw @ w1) @ w2).contiguous()
    k = (xk @ key_weight).contiguous()
    v = (xv @ value_weight).contiguous()
    a = torch.sigmoid(a0 + (xa @ a1) @ a2).contiguous()
    g = (torch.sigmoid(xg @ g1) @ g2).contiguous()

    kk = F.normalize((k * k_k).view(time_steps, heads, head_size), dim=-1, p=2.0).view(
        time_steps, heads * head_size
    )
    k = (k * (1 + (a - 1) * k_a)).contiguous()
    if layer_id == 0:
        v_first = v
    else:
        v = (v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)).contiguous()

    w = (-F.softplus(-(w0 + w)) - 0.5).contiguous()
    xx = rwkv7_op(state, r, w, k, v, (-kk).contiguous(), (kk * a).contiguous())

    xx = F.group_norm(
        xx.view(time_steps, heads * head_size),
        num_groups=heads,
        weight=ln_weight,
        bias=ln_bias,
        eps=64e-5,
    ).view(time_steps, heads * head_size)
    xx = xx + (
        (r * k * r_k).view(time_steps, heads, head_size).sum(dim=-1, keepdim=True)
        * v.view(time_steps, heads, head_size)
    ).view(time_steps, heads * head_size)
    return (xx * g) @ output_weight, x[-1, :], state, v_first


def channel_mix_seq(x, x_prev, x_k, key_weight, value_weight, semb, s1, s2, s0):
    time_steps, _ = x.shape
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
    k = x + xx * x_k
    k = torch.relu(k @ key_weight) ** 2
    ss = (x @ s1).view(time_steps, 1, 32) @ semb.view(time_steps, 32, 32)
    k = k * ((ss.view(time_steps, 32) @ s2) + s0)
    return k @ value_weight, x[-1, :]


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    env_path = os.environ.get("RWKV_MODEL_PATH") or os.environ.get("RWKV_MODEL_NAME")
    return _model_path_with_suffix(model_path or env_path or DEFAULT_MODEL_PATH)
