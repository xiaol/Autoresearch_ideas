#!/usr/bin/env python3
"""Tiny pretraining ablation for DSA/LSA/CSA/HCA-style causal attention.

This is not a reproduction of LongCat or DeepSeek training. It is a compact,
SSD-safe experiment that keeps model/data/training budget fixed and swaps only
the attention mechanism (csa/hca add a 32-param/layer comp_gate; ~0.01% of the
model — recorded in the per-variant "params" metric).

Caveats to keep the comparison honest:
- Per-query attention-slot budgets are matched at the default seq_len=256: each
  variant sees ~96 slots at the final query position (dsa: 96 oracle tokens;
  lsa: 64 local + 32 recalled tokens; csa: 64 local + top-32 of 64 4:1 blocks;
  hca: 64 local + all 32 completed 8:1 blocks). Change one knob and the
  ablation becomes a budget comparison, not a mechanism comparison.
- "dsa" is ORACLE top-k over the exact t*t score matrix — an upper bound on
  what a learned indexer (DeepSeek's lightning indexer) could select, at full
  dense cost. It measures the value of perfect sparse selection, not DSA's
  deployable mechanism.
- All variants are dense O(t^2) simulations with boolean masks; tokens_per_sec
  reflects masking overhead, NOT the efficiency of the real sparse kernels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


SSD_ROOT = Path("/run/media/xiaol/B214449214445C0B")
DEFAULT_DATA_DIR = SSD_ROOT / "autoresearch_datasets/rwkv_ms_hf_mix_5mchars"
DEFAULT_OUT_ROOT = SSD_ROOT / "attention_pretrain_ablation"
NEG_INF = -1.0e9


@dataclass
class ModelConfig:
    vocab_size: int = 257
    seq_len: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    dsa_topk: int = 96
    local_window: int = 64
    lsa_block: int = 16
    lsa_top_blocks: int = 4
    lsa_token_topk: int = 32
    csa_ratio: int = 4
    csa_topk: int = 32
    hca_ratio: int = 8

    def __post_init__(self) -> None:
        # A selection mask with an all-False row makes softmax(NEG_INF row) uniform —
        # i.e. silent attention to future tokens. Every knob below must stay >= 1 so
        # each query always keeps at least its own position.
        for field in (
            "dsa_topk",
            "local_window",
            "lsa_block",
            "lsa_top_blocks",
            "lsa_token_topk",
            "csa_ratio",
            "csa_topk",
            "hca_ratio",
        ):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be >= 1 (got {getattr(self, field)})")
        # Compressed branches only see COMPLETED blocks (block_end <= pos_q), so a
        # non-local token is covered iff ratio <= local_window + 1; otherwise tokens
        # in the query's in-progress block are invisible to both paths (no leak, but
        # a silent coverage hole that would corrupt the ablation).
        for field in ("csa_ratio", "hca_ratio"):
            if getattr(self, field) > self.local_window + 1:
                raise ValueError(
                    f"{field}={getattr(self, field)} > local_window+1={self.local_window + 1} "
                    "creates a coverage blind zone between the local window and completed blocks"
                )


def read_jsonl_text(path: Path, max_bytes: int | None = None) -> torch.Tensor:
    ids: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "")
            ids.extend(text.encode("utf-8", errors="replace"))
            ids.append(256)
            if max_bytes is not None and len(ids) >= max_bytes:
                ids = ids[:max_bytes]
                break
    return torch.tensor(ids, dtype=torch.long)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    # valid start i needs data[i + seq_len] for the target, so i <= numel - seq_len - 1
    n_starts = data.numel() - seq_len
    if n_starts <= 0:
        raise ValueError(f"dataset too small for seq_len={seq_len}: {data.numel()} tokens")
    starts = torch.randint(0, n_starts, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + seq_len] for i in starts])
    y = torch.stack([data[i + 1 : i + 1 + seq_len] for i in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def causal_mask(t: int, device: torch.device) -> torch.Tensor:
    return torch.ones((t, t), dtype=torch.bool, device=device).tril()


def masked_topk_mask(scores: torch.Tensor, allowed: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.zeros_like(allowed, dtype=torch.bool)
    k = min(k, scores.size(-1))
    masked = scores.masked_fill(~allowed, NEG_INF)
    idx = torch.topk(masked, k=k, dim=-1).indices
    out = torch.zeros_like(allowed, dtype=torch.bool)
    out.scatter_(-1, idx, True)
    return out & allowed


def split_heads(x: torch.Tensor, n_head: int) -> torch.Tensor:
    b, t, c = x.shape
    return x.view(b, t, n_head, c // n_head).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    b, h, t, d = x.shape
    return x.transpose(1, 2).contiguous().view(b, t, h * d)


def pad_to_ratio(x: torch.Tensor, ratio: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    b, h, t, d = x.shape
    n_blocks = math.ceil(t / ratio)
    padded_t = n_blocks * ratio
    pad = padded_t - t
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    valid = torch.arange(padded_t, device=x.device).view(n_blocks, ratio) < t
    return x.view(b, h, n_blocks, ratio, d), valid, n_blocks


class DSAAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.topk = cfg.dsa_topk
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, reuse_mask: torch.Tensor | None = None):
        del reuse_mask
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = split_heads(q, self.n_head), split_heads(k, self.n_head), split_heads(v, self.n_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = causal_mask(t, x.device).view(1, 1, t, t).expand(b, self.n_head, t, t)
        selected = masked_topk_mask(scores, allowed, self.topk)
        att = torch.softmax(scores.masked_fill(~selected, NEG_INF), dim=-1)
        y = torch.matmul(att, v)
        return self.drop(self.proj(merge_heads(y))), None


class LSAAttention(nn.Module):
    """LSA-inspired: local streaming window + hierarchical block recall + CLI reuse.

    The parent model reuses the boolean selection mask every two layers, which
    approximates Cross-Layer Indexing. The mask itself combines contiguous local
    access with coarse block recall and token top-k within recalled blocks.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.local_window = cfg.local_window
        self.block = cfg.lsa_block
        self.top_blocks = cfg.lsa_top_blocks
        self.token_topk = cfg.lsa_token_topk
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def build_mask(self, q: torch.Tensor, k: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        b, h, t, d = q.shape
        device = q.device
        pos_q = torch.arange(t, device=device).view(t, 1)
        pos_k = torch.arange(t, device=device).view(1, t)
        causal = pos_k <= pos_q
        local = causal & (pos_k >= (pos_q - self.local_window + 1))
        local = local.view(1, 1, t, t).expand(b, h, t, t)

        block_k, valid, n_blocks = pad_to_ratio(k, self.block)
        weights = valid.to(k.dtype).view(1, 1, n_blocks, self.block, 1)
        block_k = (block_k * weights).sum(dim=3) / weights.sum(dim=3).clamp_min(1.0)
        block_scores = torch.matmul(q, block_k.transpose(-2, -1)) / math.sqrt(d)

        block_end = (torch.arange(n_blocks, device=device) + 1) * self.block - 1
        older_allowed = block_end.view(1, n_blocks) <= (torch.arange(t, device=device).view(t, 1) - self.local_window)
        older_allowed = older_allowed.view(1, 1, t, n_blocks).expand(b, h, t, n_blocks)
        recalled_blocks = masked_topk_mask(block_scores, older_allowed, self.top_blocks)

        token_block = (torch.arange(t, device=device) // self.block).view(1, 1, 1, t).expand(b, h, t, t)
        recalled_tokens = recalled_blocks.gather(-1, token_block)
        candidates = recalled_tokens & causal.view(1, 1, t, t) & ~local
        fine = masked_topk_mask(scores, candidates, self.token_topk)
        return local | fine

    def forward(self, x: torch.Tensor, reuse_mask: torch.Tensor | None = None):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = split_heads(q, self.n_head), split_heads(k, self.n_head), split_heads(v, self.n_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        selected = reuse_mask if reuse_mask is not None else self.build_mask(q, k, scores)
        att = torch.softmax(scores.masked_fill(~selected, NEG_INF), dim=-1)
        y = torch.matmul(att, v)
        return self.drop(self.proj(merge_heads(y))), selected.detach()


class CompressedAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, *, ratio: int, sparse_topk: int | None):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.local_window = cfg.local_window
        self.ratio = ratio
        self.sparse_topk = sparse_topk
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.comp_gate = nn.Linear(self.head_dim, 1, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def compress(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        k_blocks, valid, n_blocks = pad_to_ratio(k, self.ratio)
        v_blocks, _, _ = pad_to_ratio(v, self.ratio)
        gate = self.comp_gate(k_blocks).squeeze(-1)
        gate = gate.masked_fill(~valid.view(1, 1, n_blocks, self.ratio), NEG_INF)
        w = torch.softmax(gate, dim=-1).unsqueeze(-1)
        return (k_blocks * w).sum(dim=3), (v_blocks * w).sum(dim=3), n_blocks

    def forward(self, x: torch.Tensor, reuse_mask: torch.Tensor | None = None):
        del reuse_mask
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = split_heads(q, self.n_head), split_heads(k, self.n_head), split_heads(v, self.n_head)
        k_comp, v_comp, n_blocks = self.compress(k, v)

        pos_q = torch.arange(t, device=x.device).view(t, 1)
        pos_k = torch.arange(t, device=x.device).view(1, t)
        causal = pos_k <= pos_q
        local_allowed = causal & (pos_k >= (pos_q - self.local_window + 1))
        local_allowed = local_allowed.view(1, 1, t, t).expand(b, self.n_head, t, t)

        local_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        comp_scores = torch.matmul(q, k_comp.transpose(-2, -1)) / math.sqrt(self.head_dim)
        block_end = (torch.arange(n_blocks, device=x.device) + 1) * self.ratio - 1
        comp_allowed = block_end.view(1, n_blocks) <= torch.arange(t, device=x.device).view(t, 1)
        comp_allowed = comp_allowed.view(1, 1, t, n_blocks).expand(b, self.n_head, t, n_blocks)
        if self.sparse_topk is not None:
            comp_allowed = masked_topk_mask(comp_scores, comp_allowed, self.sparse_topk)

        logits = torch.cat(
            [
                local_scores.masked_fill(~local_allowed, NEG_INF),
                comp_scores.masked_fill(~comp_allowed, NEG_INF),
            ],
            dim=-1,
        )
        att = torch.softmax(logits, dim=-1)
        att_local, att_comp = att[..., :t], att[..., t:]
        y = torch.matmul(att_local, v) + torch.matmul(att_comp, v_comp)
        return self.drop(self.proj(merge_heads(y))), None


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, variant: str):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        if variant == "dsa":
            self.attn = DSAAttention(cfg)
        elif variant == "lsa":
            self.attn = LSAAttention(cfg)
        elif variant == "csa":
            self.attn = CompressedAttention(cfg, ratio=cfg.csa_ratio, sparse_topk=cfg.csa_topk)
        elif variant == "hca":
            self.attn = CompressedAttention(cfg, ratio=cfg.hca_ratio, sparse_topk=None)
        else:
            raise ValueError(f"unknown variant: {variant}")
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, reuse_mask: torch.Tensor | None = None):
        y, selected = self.attn(self.ln1(x), reuse_mask)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, selected


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig, variant: str):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, variant) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos).view(1, t, -1)
        reuse_mask = None
        for layer_id, block in enumerate(self.blocks):
            if self.variant == "lsa":
                x, selected = block(x, reuse_mask)
                reuse_mask = selected if layer_id % 2 == 0 else None
            else:
                x, _ = block(x, None)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    eval_iters: int,
    use_amp: bool,
) -> float:
    model.eval()
    losses = []
    gen = torch.Generator().manual_seed(12345)
    for _ in range(eval_iters):
        xb, yb = get_batch(data, batch_size, seq_len, device, gen)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(xb, yb)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def train_variant(
    variant: str,
    cfg: ModelConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    model = TinyLM(cfg, variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch_gen = torch.Generator().manual_seed(args.seed + 999)
    metrics_path = out_dir / variant / "metrics.jsonl"
    (out_dir / variant).mkdir(parents=True, exist_ok=True)
    metrics_path.unlink(missing_ok=True)  # a rerun with the same --run-name must not interleave rows

    n_params = sum(p.numel() for p in model.parameters())
    best_val = float("inf")
    last_metrics: dict = {}
    start = time.time()
    eval_seconds = 0.0
    for step in range(1, args.steps + 1):
        xb, yb = get_batch(train_data, args.batch_size, cfg.seq_len, device, batch_gen)
        lr = args.lr
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            _, loss = model(xb, yb)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        train_elapsed = elapsed - eval_seconds
        tokens = step * args.batch_size * cfg.seq_len
        train_loss = float(loss.item())
        should_eval = step == 1 or step % args.eval_interval == 0 or step == args.steps
        if should_eval:
            eval_start = time.time()
            val_loss = estimate_loss(
                model,
                val_data,
                args.eval_batch_size,
                cfg.seq_len,
                device,
                args.eval_iters,
                args.amp and device.type == "cuda",
            )
            eval_seconds += time.time() - eval_start
            best_val = min(best_val, val_loss)
            last_metrics = {
                "variant": variant,
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val,
                "elapsed_sec": elapsed,
                "train_sec": train_elapsed,
                "tokens": tokens,
                "tokens_per_sec": tokens / max(train_elapsed, 1e-9),
                "params": n_params,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(last_metrics) + "\n")
            print(
                f"{variant:>3} step {step:4d}/{args.steps} "
                f"train {train_loss:.4f} val {val_loss:.4f} "
                f"tok/s {last_metrics['tokens_per_sec']:.0f}",
                flush=True,
            )

    ckpt = {
        "variant": variant,
        "model_config": asdict(cfg),
        "args": vars(args),
        "model": model.state_dict(),
        "last_metrics": last_metrics,
    }
    torch.save(ckpt, out_dir / variant / "final.pt")
    return last_metrics


def parse_variants(raw: str) -> list[str]:
    variants = [v.strip().lower() for v in raw.split(",") if v.strip()]
    allowed = {"dsa", "lsa", "csa", "hca"}
    bad = [v for v in variants if v not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown variants {bad}; allowed={sorted(allowed)}")
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--variants", type=parse_variants, default=parse_variants("dsa,lsa,csa,hca"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-iters", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dsa-topk", type=int, default=ModelConfig.dsa_topk)
    parser.add_argument("--local-window", type=int, default=ModelConfig.local_window)
    parser.add_argument("--lsa-block", type=int, default=ModelConfig.lsa_block)
    parser.add_argument("--lsa-top-blocks", type=int, default=ModelConfig.lsa_top_blocks)
    parser.add_argument("--lsa-token-topk", type=int, default=ModelConfig.lsa_token_topk)
    parser.add_argument("--csa-ratio", type=int, default=ModelConfig.csa_ratio)
    parser.add_argument("--csa-topk", type=int, default=ModelConfig.csa_topk)
    parser.add_argument("--hca-ratio", type=int, default=ModelConfig.hca_ratio)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-bytes", type=int, default=None)
    args = parser.parse_args()

    cfg = ModelConfig(
        seq_len=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dsa_topk=args.dsa_topk,
        local_window=args.local_window,
        lsa_block=args.lsa_block,
        lsa_top_blocks=args.lsa_top_blocks,
        lsa_token_topk=args.lsa_token_topk,
        csa_ratio=args.csa_ratio,
        csa_topk=args.csa_topk,
        hca_ratio=args.hca_ratio,
    )
    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError("--n-embd must be divisible by --n-head")

    out_dir = args.out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "train.jsonl"
    val_path = args.data_dir / "validation.jsonl"
    train_data = read_jsonl_text(train_path, args.max_train_bytes)
    val_data = read_jsonl_text(val_path, None)

    config_blob = {
        "model_config": asdict(cfg),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_tokens": int(train_data.numel()),
        "val_tokens": int(val_data.numel()),
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_blob, f, indent=2)

    print(f"output: {out_dir}")
    print(f"train tokens: {train_data.numel():,}; val tokens: {val_data.numel():,}")
    print(f"device: {args.device}; variants: {','.join(args.variants)}")

    results = []
    for variant in args.variants:
        results.append(train_variant(variant, cfg, train_data, val_data, args, out_dir))

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"summary: {summary_path}")
    for row in sorted(results, key=lambda r: r.get("best_val_loss", float("inf"))):
        print(
            f"{row['variant']:>3} best_val={row['best_val_loss']:.4f} "
            f"last_val={row['val_loss']:.4f} tok/s={row['tokens_per_sec']:.0f}"
        )


if __name__ == "__main__":
    main()
