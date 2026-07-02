#!/usr/bin/env python3
"""Empirical correctness checks for train_attention_variants.py attention code.

Checks, per variant (dsa/lsa/csa/hca):
  1. Gradient causality on the raw attention module: d y[i] / d x[j] == 0 for j > i.
  2. Full-model perturbation causality: changing token j leaves logits[:, <j] unchanged.
  3. Mask sanity (LSA/DSA): selection is a subset of the causal mask; every row non-empty.
  4. Degenerate-equivalence: DSA with topk>=t and LSA with local_window>=t must equal
     dense causal softmax attention with the same weights.
  5. Odd sequence lengths (t not divisible by block/ratio): finite outputs, causality holds.
  6. One training-style step: finite loss and gradients.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/xiaol/X/attention_pretrain_ablation")
from train_attention_variants import (  # noqa: E402
    CompressedAttention,
    DSAAttention,
    LSAAttention,
    ModelConfig,
    TinyLM,
    causal_mask,
    split_heads,
)

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(f"{name} :: {detail}")


def make_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=257, seq_len=256, n_layer=4, n_head=4, n_embd=64)
    base.update(kw)
    return ModelConfig(**base)


def build_attn(variant: str, cfg: ModelConfig):
    if variant == "dsa":
        return DSAAttention(cfg)
    if variant == "lsa":
        return LSAAttention(cfg)
    if variant == "csa":
        return CompressedAttention(cfg, ratio=cfg.csa_ratio, sparse_topk=cfg.csa_topk)
    if variant == "hca":
        return CompressedAttention(cfg, ratio=cfg.hca_ratio, sparse_topk=None)
    raise ValueError(variant)


def grad_causality(variant: str, t: int) -> None:
    cfg = make_cfg()
    attn = build_attn(variant, cfg)
    x = torch.randn(2, t, cfg.n_embd, requires_grad=True)
    y, _ = attn(x)
    check(f"{variant} t={t} output finite", bool(torch.isfinite(y).all()))
    for i in (0, t // 2, t - 2):
        g = torch.autograd.grad(y[:, i].sum(), x, retain_graph=True)[0]
        leak = g[:, i + 1 :].abs().max().item() if i + 1 < t else 0.0
        check(f"{variant} t={t} grad-causality at pos {i}", leak == 0.0, f"future-grad={leak:.3e}")


def model_perturbation(variant: str, t: int) -> None:
    cfg = make_cfg(seq_len=max(t, 64))
    model = TinyLM(cfg, variant).eval()
    idx = torch.randint(0, 257, (2, t))
    with torch.no_grad():
        base, _ = model(idx)
        j = t // 2
        idx2 = idx.clone()
        idx2[:, j] = (idx2[:, j] + 1) % 257
        pert, _ = model(idx2)
    diff = (base[:, :j] - pert[:, :j]).abs().max().item()
    check(f"{variant} t={t} full-model causality (perturb pos {j})", diff == 0.0, f"past-logit-diff={diff:.3e}")
    check(f"{variant} t={t} full-model logits finite", bool(torch.isfinite(base).all()))


def dense_reference(q, k, v, head_dim):
    t = q.shape[2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    scores = scores.masked_fill(~causal_mask(t, q.device), float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def degenerate_equivalence() -> None:
    t = 48
    # DSA with topk >= t must be dense causal attention.
    cfg = make_cfg(dsa_topk=t)
    attn = build_attn("dsa", cfg)
    x = torch.randn(2, t, cfg.n_embd)
    q, k, v = attn.qkv(x).chunk(3, dim=-1)
    q, k, v = (split_heads(z, cfg.n_head) for z in (q, k, v))
    ref = dense_reference(q, k, v, attn.head_dim)
    y, _ = attn(x)
    got = y  # proj+drop applied; apply same to ref
    ref_out = attn.drop(attn.proj(ref.transpose(1, 2).contiguous().view(2, t, -1)))
    diff = (got - ref_out).abs().max().item()
    check("dsa topk>=t equals dense causal attention", diff < 1e-10, f"max-diff={diff:.3e}")

    # LSA with local_window >= t must be dense causal attention.
    cfg = make_cfg(local_window=t)
    attn = build_attn("lsa", cfg)
    q, k, v = attn.qkv(x).chunk(3, dim=-1)
    q, k, v = (split_heads(z, cfg.n_head) for z in (q, k, v))
    ref = dense_reference(q, k, v, attn.head_dim)
    y, sel = attn(x)
    ref_out = attn.drop(attn.proj(ref.transpose(1, 2).contiguous().view(2, t, -1)))
    diff = (y - ref_out).abs().max().item()
    check("lsa window>=t equals dense causal attention", diff < 1e-10, f"max-diff={diff:.3e}")
    check("lsa window>=t mask == causal", bool((sel == causal_mask(t, x.device)).all()))


def mask_sanity() -> None:
    t = 200  # not divisible by lsa_block=16 -> exercises padding path
    cfg = make_cfg(local_window=32, lsa_block=16, lsa_top_blocks=3, lsa_token_topk=8)
    attn = build_attn("lsa", cfg)
    x = torch.randn(2, t, cfg.n_embd)
    _, sel = attn(x)
    causal = causal_mask(t, x.device)
    check("lsa mask subset of causal", bool((sel & ~causal).sum() == 0))
    check("lsa mask rows non-empty", bool(sel.any(dim=-1).all()))
    # local window fully included
    pos_q = torch.arange(t).view(t, 1)
    pos_k = torch.arange(t).view(1, t)
    local = (pos_k <= pos_q) & (pos_k >= pos_q - cfg.local_window + 1)
    check("lsa local window fully selected", bool((sel & local.view(1, 1, t, t) == local.view(1, 1, t, t)).all()))
    # recalled (non-local) token budget per row <= token_topk
    fine = sel & ~local.view(1, 1, t, t)
    check("lsa recall budget respected", int(fine.sum(-1).max()) <= cfg.lsa_token_topk)
    # recalled tokens only in fully-older blocks
    blk = pos_k // cfg.lsa_block
    blk_end = (blk + 1) * cfg.lsa_block - 1
    older = blk_end <= (pos_q - cfg.local_window)
    check("lsa recalled tokens are in fully-older blocks", bool((fine & ~older.view(1, 1, t, t)).sum() == 0))

    cfg = make_cfg(dsa_topk=16)
    attn = build_attn("dsa", cfg)
    scores = torch.randn(2, cfg.n_head, t, t)
    from train_attention_variants import masked_topk_mask

    allowed = causal_mask(t, scores.device).view(1, 1, t, t).expand_as(scores)
    sel = masked_topk_mask(scores, allowed, 16)
    check("dsa topk mask subset of causal", bool((sel & ~allowed).sum() == 0))
    check("dsa topk row budget", int(sel.sum(-1).max()) <= 16)
    check("dsa topk early rows = all allowed", bool((sel[..., :16, :] == allowed[..., :16, :]).all()))


def odd_lengths() -> None:
    for variant, t in (("csa", 101), ("hca", 77), ("lsa", 101), ("dsa", 33)):
        grad_causality(variant, t)
        model_perturbation(variant, t)


def train_step() -> None:
    for variant in ("dsa", "lsa", "csa", "hca"):
        cfg = make_cfg(seq_len=64)
        model = TinyLM(cfg, variant)
        idx = torch.randint(0, 257, (4, 64))
        tgt = torch.randint(0, 257, (4, 64))
        _, loss = model(idx, tgt)
        loss.backward()
        finite = torch.isfinite(loss).item() and all(
            torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
        )
        check(f"{variant} train step loss/grads finite", bool(finite), f"loss={loss.item():.4f}")


def harness_regressions() -> None:
    from train_attention_variants import get_batch

    # minimal dataset (numel == seq_len + 1) has exactly one valid window and must work
    data = torch.arange(9, dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    x, y = get_batch(data, 4, 8, torch.device("cpu"), gen)
    check("get_batch accepts minimal dataset", True)
    check("get_batch y is x shifted by one", bool((y[:, :-1] == x[:, 1:]).all()))
    # every valid start (only 0 here) reachable; last token appears as a target
    check("get_batch reaches last token as target", int(y.max()) == 8)

    # degenerate sparsity configs must be rejected, not silently leak future tokens
    for field in ("dsa_topk", "local_window", "csa_topk", "hca_ratio"):
        try:
            make_cfg(**{field: 0})
            check(f"ModelConfig rejects {field}=0", False, "no ValueError raised")
        except ValueError:
            check(f"ModelConfig rejects {field}=0", True)


def main() -> None:
    for variant in ("dsa", "lsa", "csa", "hca"):
        grad_causality(variant, 128)
        model_perturbation(variant, 128)
    degenerate_equivalence()
    mask_sanity()
    odd_lengths()
    train_step()
    harness_regressions()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
