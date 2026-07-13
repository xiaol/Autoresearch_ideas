"""Fit a Jacobian Lens artifact for an RWKV-7 G1 checkpoint.

The production RWKV wrapper uses a forward-only recurrent CUDA kernel. Fitting
needs gradients, so this module mirrors the same recurrence with ordinary
PyTorch operations. Its production estimator freezes prior-token state to fit
a same-position Jacobian and affine residual centers. The generated checkpoint
is consumed by ``rwkv_jlens_adapter.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


DEFAULT_MODEL_PATH = "/home/xiaol/X/models/rwkv7-g1/rwkv7a-g1d-0.1b-20260212-ctx8192.pth"
DEFAULT_RWKV_SOURCE = "/home/xiaol/X/rwkv-manifold-steering/src"
DEFAULT_CALIBRATION_PROMPTS = (
    "Scientific progress depends on measurements that can be repeated, theories that make "
    "risky predictions, and experiments designed to distinguish competing explanations.",
    "Reliable software has clear interfaces, explicit failure modes, focused tests, and logs "
    "that make unexpected behavior possible to diagnose.",
    "A city connects transport, water, energy, housing, schools, hospitals, and communications "
    "through decisions made across many timescales.",
    "Historical evidence comes from documents, artifacts, oral accounts, and the careful "
    "comparison of sources that may disagree with one another.",
    "In mathematics, a proof explains why a statement follows from definitions and earlier "
    "results, including edge cases that examples might miss.",
    "A healthy ecosystem contains many interacting species, flows of energy, nutrient cycles, "
    "and feedback mechanisms that respond to environmental change.",
    "Good writing gives each paragraph a purpose, uses concrete evidence, and guides the reader "
    "through an argument without hiding important qualifications.",
    "Economic choices involve incentives, scarce resources, uncertainty, institutions, and "
    "effects that often appear far from the original transaction.",
    "Learning a language requires repeated exposure to meaningful speech, attention to patterns, "
    "and enough practice to retrieve words under real conditions.",
    "A medical diagnosis combines symptoms, examination findings, test results, prior "
    "probabilities, and the costs of missing alternative explanations.",
    "Computer networks divide information into packets, route them across imperfect links, "
    "detect failures, and retry work when delivery is uncertain.",
    "Music creates expectation through rhythm, melody, harmony, timbre, repetition, and carefully "
    "timed departures from patterns already established.",
    "Legal reasoning applies general rules to particular facts while considering precedent, "
    "procedure, competing rights, and the practical meaning of remedies.",
    "A good teacher checks what students already understand, selects examples that reveal "
    "structure, and adjusts explanations when misconceptions appear.",
    "Climate systems couple the atmosphere, oceans, ice, land, and living organisms through "
    "feedback loops operating over different spatial scales.",
    "Engineering design turns requirements into tradeoffs among safety, cost, performance, "
    "maintainability, and the uncertainty of real operating conditions.",
    "A database transaction groups related changes so concurrent users observe consistent state "
    "even when machines fail or requests arrive simultaneously.",
    "Astronomers infer distant objects from light, motion, spectra, and statistical models because "
    "direct experiments on stars and galaxies are impossible.",
    "Cooking transforms ingredients through heat, mixing, fermentation, timing, and chemical "
    "reactions whose effects depend on proportion and sequence.",
    "Public policy should state its objective, identify who bears costs and benefits, measure "
    "outcomes, and change when evidence contradicts assumptions.",
    "Cryptographic systems rely on precise threat models, well studied algorithms, protected "
    "keys, and implementations that do not leak secrets indirectly.",
    "Evolution changes populations across generations as mutation, selection, drift, migration, "
    "and environmental pressures alter inherited variation.",
    "A map simplifies the world for a purpose, choosing which distances, boundaries, landmarks, "
    "and relationships deserve visual emphasis.",
    "Journalism serves readers by verifying claims, separating observation from inference, "
    "naming uncertainty, and correcting errors in a visible way.",
    "Robotics combines sensing, estimation, planning, control, and mechanical design so a machine "
    "can act despite noise and incomplete information.",
    "Philosophical arguments become clearer when premises are explicit, terms remain consistent, "
    "counterexamples are taken seriously, and conclusions are limited.",
    "Agriculture depends on soil, water, weather, genetics, labor, markets, and long planning "
    "cycles in which short term gains can create later costs.",
    "Cybersecurity is a continuous process of reducing exposure, detecting suspicious behavior, "
    "containing incidents, and learning from attempted attacks.",
    "Architecture organizes structure, circulation, light, materials, climate, and human activity "
    "into spaces that must work over many years.",
    "Statistical inference uses samples to reason about a larger population while quantifying "
    "uncertainty and checking whether modeling assumptions are plausible.",
    "A team coordinates effectively when responsibilities are visible, decisions have owners, "
    "disagreements use evidence, and important context is written down.",
    "Memory is reconstructive rather than a perfect recording, so recall depends on attention, "
    "context, emotion, rehearsal, and cues available later.",
)


def default_output_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    return str(path.with_name(f"{path.stem}_jacobian_lens.pt"))


def sha256_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_layers(value: str | None, n_layer: int) -> list[int]:
    if not value:
        return list(range(n_layer - 1))
    layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not layers or layers[0] < 0 or layers[-1] >= n_layer - 1:
        raise ValueError(f"source layers must be in [0, {n_layer - 2}], got {layers}")
    return layers


class DifferentiableRWKV7:
    """Gradient-enabled mirror of ``RWKV7AModel._forward_seq``."""

    def __init__(self, model: Any, *, detach_cross_position: bool = False) -> None:
        self.model = model
        self.z = model.z
        self.device = model.device
        self.n_layer = int(model.n_layer)
        self.d_model = int(model.n_embd)
        self.n_head = int(model.n_head)
        self.head_size = int(model.head_size)
        self.detach_cross_position = detach_cross_position

    def forward_residuals(
        self,
        token_ids: list[int],
        *,
        batch_size: int,
        source_layers: list[int],
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        if not token_ids:
            raise ValueError("token_ids cannot be empty")
        first_source = min(source_layers)
        idx = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        x = self.z["emb.weight"][idx].unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        captures: dict[int, torch.Tensor] = {}
        v_first: torch.Tensor | None = None

        for layer in range(self.n_layer):
            block = f"blocks.{layer}."
            att = f"{block}att."
            ffn = f"{block}ffn."

            xx = F.layer_norm(
                x,
                (self.d_model,),
                weight=self.z[f"{block}ln1.weight"],
                bias=self.z[f"{block}ln1.bias"],
            )
            att_out, v_first = self._time_mix(layer, xx, v_first, att)
            x = x + att_out

            xx = F.layer_norm(
                x,
                (self.d_model,),
                weight=self.z[f"{block}ln2.weight"],
                bias=self.z[f"{block}ln2.bias"],
            )
            x = x + self._channel_mix(xx, idx, ffn)

            if layer == first_source:
                x = x.detach().requires_grad_(True)
            if layer in source_layers:
                captures[layer] = x

        return captures, x

    def logits(self, residual: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(
            residual,
            (self.d_model,),
            weight=self.z["ln_out.weight"],
            bias=self.z["ln_out.bias"],
        )
        return x @ self.z["head.weight"]

    def _time_mix(
        self,
        layer: int,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
        prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, channels = x.shape
        zeros = torch.zeros((batch, 1, channels), device=x.device, dtype=x.dtype)
        previous = x[:, :-1, :]
        if self.detach_cross_position:
            previous = previous.detach()
        delta = torch.cat((zeros, previous), dim=1) - x
        xr = x + delta * self.z[f"{prefix}x_r"]
        xw = x + delta * self.z[f"{prefix}x_w"]
        xk = x + delta * self.z[f"{prefix}x_k"]
        xv = x + delta * self.z[f"{prefix}x_v"]
        xa = x + delta * self.z[f"{prefix}x_a"]
        xg = x + delta * self.z[f"{prefix}x_g"]

        r = xr @ self.z[f"{prefix}receptance.weight"]
        w = -F.softplus(
            -(self.z[f"{prefix}w0"] + (torch.tanh(xw @ self.z[f"{prefix}w1"]) @ self.z[f"{prefix}w2"]))
        ) - 0.5
        k = xk @ self.z[f"{prefix}key.weight"]
        v = xv @ self.z[f"{prefix}value.weight"]
        a = torch.sigmoid(
            self.z[f"{prefix}a0"] + (xa @ self.z[f"{prefix}a1"]) @ self.z[f"{prefix}a2"]
        )
        g = torch.sigmoid(xg @ self.z[f"{prefix}g1"]) @ self.z[f"{prefix}g2"]

        kk = F.normalize(
            (k * self.z[f"{prefix}k_k"]).view(batch, seq_len, self.n_head, self.head_size),
            dim=-1,
            p=2.0,
        ).view(batch, seq_len, channels)
        k = k * (1 + (a - 1) * self.z[f"{prefix}k_a"])
        if layer == 0:
            v_first = v
        else:
            if v_first is None:
                raise RuntimeError("layer-0 value residual was not initialized")
            v = v + (v_first - v) * torch.sigmoid(
                self.z[f"{prefix}v0"] + (xv @ self.z[f"{prefix}v1"]) @ self.z[f"{prefix}v2"]
            )

        recurrent = self._wkv_recurrence(r, w, k, v, -kk, kk * a)
        recurrent = F.group_norm(
            recurrent.reshape(batch * seq_len, channels),
            num_groups=self.n_head,
            weight=self.z[f"{prefix}ln_x.weight"],
            bias=self.z[f"{prefix}ln_x.bias"],
            eps=64e-5,
        ).view(batch, seq_len, channels)
        recurrent = recurrent + (
            (
                r.view(batch, seq_len, self.n_head, self.head_size)
                * k.view(batch, seq_len, self.n_head, self.head_size)
                * self.z[f"{prefix}r_k"].view(1, 1, self.n_head, self.head_size)
            ).sum(dim=-1, keepdim=True)
            * v.view(batch, seq_len, self.n_head, self.head_size)
        ).view(batch, seq_len, channels)
        return (recurrent * g) @ self.z[f"{prefix}output.weight"], v_first

    def _wkv_recurrence(
        self,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = r.shape
        shape = (batch, self.n_head, self.head_size, self.head_size)
        state = torch.zeros(shape, device=r.device, dtype=torch.float32)
        outputs: list[torch.Tensor] = []

        r_h = r.view(batch, seq_len, self.n_head, self.head_size).float()
        k_h = k.view(batch, seq_len, self.n_head, self.head_size).float()
        v_h = v.view(batch, seq_len, self.n_head, self.head_size).float()
        a_h = a.view(batch, seq_len, self.n_head, self.head_size).float()
        b_h = b.view(batch, seq_len, self.n_head, self.head_size).float()
        decay = torch.exp(-torch.exp(w.float())).view(batch, seq_len, self.n_head, self.head_size)

        for position in range(seq_len):
            state_times_a = torch.einsum("bhij,bhj->bhi", state, a_h[:, position])
            state = (
                state * decay[:, position, :, None, :]
                + torch.einsum("bhi,bhj->bhij", v_h[:, position], k_h[:, position])
                + torch.einsum("bhi,bhj->bhij", state_times_a, b_h[:, position])
            )
            outputs.append(torch.einsum("bhij,bhj->bhi", state, r_h[:, position]))
            if self.detach_cross_position:
                # Preserve the exact recurrent value while treating history as
                # fixed for the next token's same-position local Jacobian.
                state = state.detach()

        return torch.stack(outputs, dim=1).reshape(batch, seq_len, -1).to(dtype=r.dtype)

    def _channel_mix(self, x: torch.Tensor, idx: torch.Tensor, prefix: str) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        zeros = torch.zeros((batch, 1, channels), device=x.device, dtype=x.dtype)
        previous = x[:, :-1, :]
        if self.detach_cross_position:
            previous = previous.detach()
        delta = torch.cat((zeros, previous), dim=1) - x
        k = x + delta * self.z[f"{prefix}x_k"]
        k = torch.relu(k @ self.z[f"{prefix}key.weight"]) ** 2
        semb = self.z[f"{prefix}s_emb.weight"][idx].view(seq_len, 32, 32)
        ss = torch.matmul((x @ self.z[f"{prefix}s1"]).view(batch, seq_len, 1, 32), semb)
        scale = ss.view(batch, seq_len, 32) @ self.z[f"{prefix}s2"] + self.z[f"{prefix}s0"]
        return (k * scale) @ self.z[f"{prefix}value.weight"]


def parity_check(
    model: Any,
    differentiable: DifferentiableRWKV7,
    token_ids: list[int],
    *,
    min_cosine: float = 0.999,
) -> tuple[float, float, float]:
    with torch.inference_mode():
        expected_output = model.forward(token_ids, full_output=True, collect_layers=True)
        expected = expected_output.logits.float()
    with torch.enable_grad():
        captures, residual = differentiable.forward_residuals(
            token_ids,
            batch_size=1,
            source_layers=list(range(differentiable.n_layer)),
        )
    with torch.no_grad():
        actual = differentiable.logits(residual).squeeze(0).float()
        hidden_max_abs = max(
            float(
                (
                    expected_output.hidden_by_layer[layer].float()
                    - captures[layer][0, -1].detach().float().cpu()
                )
                .abs()
                .max()
            )
            for layer in range(differentiable.n_layer)
        )
        cosine = float(F.cosine_similarity(expected.reshape(-1), actual.reshape(-1), dim=0))
        max_abs = float((expected - actual).abs().max())
    if not math.isfinite(cosine) or cosine < min_cosine:
        raise RuntimeError(
            f"differentiable RWKV parity check failed: cosine={cosine:.8f}, max_abs={max_abs:.6g}"
        )
    return cosine, max_abs, hidden_max_abs


def jacobians_for_tokens(
    differentiable: DifferentiableRWKV7,
    token_ids: list[int],
    source_layers: list[int],
    *,
    dim_batch: int,
    skip_first: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
    seq_len = len(token_ids)
    if seq_len <= skip_first + 1:
        raise ValueError(f"sequence length {seq_len} must be greater than skip_first + 1 ({skip_first + 1})")
    batch_size = min(dim_batch, differentiable.d_model)
    captures, target = differentiable.forward_residuals(
        token_ids,
        batch_size=batch_size,
        source_layers=source_layers,
    )
    sources = [captures[layer] for layer in source_layers]
    valid_positions = torch.arange(skip_first, seq_len - 1, device=target.device)
    source_means = {
        layer: captures[layer][0, valid_positions].detach().float().mean(dim=0).cpu()
        for layer in source_layers
    }
    target_mean = target[0, valid_positions].detach().float().mean(dim=0).cpu()
    jacobians = {
        layer: torch.empty(
            differentiable.d_model,
            differentiable.d_model,
            dtype=torch.float32,
        )
        for layer in source_layers
    }
    passes = math.ceil(differentiable.d_model / batch_size)
    cotangent = torch.zeros_like(target)

    for pass_index, dim_start in enumerate(range(0, differentiable.d_model, batch_size)):
        dims_this_pass = min(batch_size, differentiable.d_model - dim_start)
        cotangent.zero_()
        batch_indices = torch.arange(dims_this_pass, device=target.device)
        cotangent[
            batch_indices[:, None],
            valid_positions[None, :],
            dim_start + batch_indices[:, None],
        ] = 1
        grads = torch.autograd.grad(
            outputs=target,
            inputs=sources,
            grad_outputs=cotangent,
            retain_graph=pass_index < passes - 1,
        )
        for layer, grad in zip(source_layers, grads, strict=True):
            rows = grad[:dims_this_pass, valid_positions, :].mean(dim=1).float().cpu()
            jacobians[layer][dim_start : dim_start + dims_this_pass] = rows

        print(
            f"  dimensions {dim_start + dims_this_pass:4d}/{differentiable.d_model}",
            end="\r" if pass_index < passes - 1 else "\n",
            flush=True,
        )

    return jacobians, source_means, target_mean


def fit(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    source_layers: list[int],
    *,
    max_seq_len: int,
    dim_batch: int,
    skip_first: int,
    estimator: str,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
    differentiable = DifferentiableRWKV7(
        model,
        detach_cross_position=estimator == "same_position_mean",
    )
    sums = {
        layer: torch.zeros(differentiable.d_model, differentiable.d_model, dtype=torch.float32)
        for layer in source_layers
    }
    source_mean_sums = {
        layer: torch.zeros(differentiable.d_model, dtype=torch.float32)
        for layer in source_layers
    }
    target_mean_sum = torch.zeros(differentiable.d_model, dtype=torch.float32)

    for prompt_index, prompt in enumerate(prompts, start=1):
        token_ids = list(tokenizer.encode(prompt))[:max_seq_len]
        if prompt_index == 1:
            cosine, max_abs, hidden_max_abs = parity_check(model, differentiable, token_ids)
            print(
                f"CUDA parity: cosine={cosine:.8f}, logits_max_abs={max_abs:.6g}, "
                f"hidden_max_abs={hidden_max_abs:.6g}"
            )
        started = time.perf_counter()
        print(f"Prompt {prompt_index}/{len(prompts)}: {len(token_ids)} tokens")
        current, current_source_means, current_target_mean = jacobians_for_tokens(
            differentiable,
            token_ids,
            source_layers,
            dim_batch=dim_batch,
            skip_first=skip_first,
        )
        for layer in source_layers:
            sums[layer].add_(current[layer])
            source_mean_sums[layer].add_(current_source_means[layer])
        target_mean_sum.add_(current_target_mean)
        print(f"  elapsed={time.perf_counter() - started:.1f}s")

    return (
        {layer: matrix / len(prompts) for layer, matrix in sums.items()},
        {layer: vector / len(prompts) for layer, vector in source_mean_sums.items()},
        target_mean_sum / len(prompts),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("RWKV_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--rwkv-source", default=os.environ.get("RWKV_SOURCE", DEFAULT_RWKV_SOURCE))
    parser.add_argument("--output", default=os.environ.get("RWKV_JLENS_PATH"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-layers", default=None, help="comma-separated layers; default is all but final")
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--dim-batch", type=int, default=4)
    parser.add_argument("--skip-first", type=int, default=8)
    parser.add_argument(
        "--estimator",
        choices=("same_position_mean", "cross_position_mean"),
        default="same_position_mean",
        help=(
            "same_position_mean freezes prior-token recurrent state while differentiating; "
            "cross_position_mean reproduces the generic Transformer estimator for diagnostics "
            "but is rejected by the production RWKV adapter"
        ),
    )
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--parity-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise SystemExit("RWKV Jacobian fitting currently requires CUDA")
    if int(args.dim_batch) < 1:
        raise SystemExit("--dim-batch must be at least 1")

    rwkv_source = str(Path(args.rwkv_source).expanduser())
    if rwkv_source not in sys.path:
        sys.path.insert(0, rwkv_source)
    from rwkv_manifold_steering.rwkv7a_model import RWKV7AModel
    from rwkv_manifold_steering.tokenizer import RWKVTokenizer

    model_path = str(Path(args.model_path).expanduser())
    output = str(Path(args.output or default_output_path(model_path)).expanduser())
    prompts = list(args.prompts or DEFAULT_CALIBRATION_PROMPTS)
    print(f"Loading RWKV checkpoint: {model_path}")
    model = RWKV7AModel(model_path, device=args.device, compile_cuda=True)
    tokenizer = RWKVTokenizer()
    source_layers = parse_layers(args.source_layers, int(model.n_layer))
    if args.parity_only:
        differentiable = DifferentiableRWKV7(
            model,
            detach_cross_position=args.estimator == "same_position_mean",
        )
        token_ids = list(tokenizer.encode(prompts[0]))[: int(args.max_seq_len)]
        cosine, max_abs, hidden_max_abs = parity_check(model, differentiable, token_ids)
        print(
            f"CUDA parity: cosine={cosine:.8f}, logits_max_abs={max_abs:.6g}, "
            f"hidden_max_abs={hidden_max_abs:.6g}"
        )
        return
    jacobians, source_means, target_mean = fit(
        model,
        tokenizer,
        prompts,
        source_layers,
        max_seq_len=int(args.max_seq_len),
        dim_batch=int(args.dim_batch),
        skip_first=int(args.skip_first),
        estimator=str(args.estimator),
    )

    prompt_hash = hashlib.sha256("\n\n".join(prompts).encode("utf-8")).hexdigest()
    checkpoint = {
        "J": {layer: matrix.to(torch.float16) for layer, matrix in jacobians.items()},
        "source_means": {
            layer: vector.to(torch.float16) for layer, vector in source_means.items()
        },
        "target_mean": target_mean.to(torch.float16),
        "format_version": 2,
        "n_prompts": len(prompts),
        "source_layers": source_layers,
        "d_model": int(model.n_embd),
        "n_layer": int(model.n_layer),
        "architecture": "rwkv7-g1",
        "activation_site": "block_output",
        "transport": "affine_centered",
        "target_layer": int(model.n_layer) - 1,
        "estimator": str(args.estimator),
        "tokenizer": "rwkv_vocab_v20230424",
        "model_path": model_path,
        "model_sha256": sha256_file(model_path),
        "calibration_sha256": prompt_hash,
        "max_seq_len": int(args.max_seq_len),
        "skip_first": int(args.skip_first),
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(destination)
    print(f"Saved RWKV Jacobian Lens ({len(source_layers)} layers) -> {destination}")


if __name__ == "__main__":
    main()
