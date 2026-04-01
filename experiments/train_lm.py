from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from auto_research_llm_ideas.experiments.data import default_device, iter_eval_batches, load_byte_corpus, random_lm_batch, set_seed
from auto_research_llm_ideas.model import (
    RecurrentFFNConfig,
    RecurrentFFNLM,
    TransformerConfig,
    TransformerLM,
    TripleLatentLM,
    UniMatrixLM,
    triple_latent_config,
    variant_config,
)


GENERIC_LM_MODELS = [
    "transformer",
    "transformer_gelu",
    "transformer_swiglu_matched",
    "recurrent_ffn",
    "recurrent_ffn_readout",
    "recurrent_ffn_hybrid",
    "recurrent_ffn_matrix",
    "recurrent_ffn_multiscale",
    "recurrent_ffn_predictive",
    "recurrent_ffn_stable",
    "recurrent_ffn_ultra",
    "recurrent_ffn_sparse",
    "recurrent_ffn_aux",
    "unimatrix-core",
    "unimatrix-rosa",
    "unimatrix-discovery",
    "unimatrix-assoc",
    "unimatrix-assoc-hard",
    "unimatrix-rosa-assoc",
    "unimatrix-rosa-assoc-hard",
    "unimatrix-discovery-assoc",
    "unimatrix-sparsepointer",
    "triple-latent",
    "triple-slot",
    "triple-hybrid",
    "triple-slot-hybrid",
    "triple-hybrid-assoc-sum",
    "triple-hybrid-assoc-concat",
    "triple-hybrid-assoc-logits",
    "triple-hybrid-assoc-serial-sum",
    "triple-hybrid-assoc-last-sum",
    "triple-hybrid-assoc-last-logits",
    "triple-hybrid-assoc-prev-sum",
    "triple-hybrid-assoc-prev-logits",
    "triple-hybrid-assoc-gated-last-logits",
    "triple-hybrid-assoc-writegated-last-logits",
    "triple-hybrid-assoc-writegated-last-sum",
    "triple-hybrid-assoc-writegated-all-logits",
]


def build_model(
    model_name: str,
    vocab_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    dropout: float,
    pad_token_id: int,
) -> torch.nn.Module:
    if model_name == "transformer":
        cfg = TransformerConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=seq_len,
            ffn_type="swiglu",
        )
        return TransformerLM(cfg)

    if model_name == "transformer_gelu":
        cfg = TransformerConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=seq_len,
            ffn_type="gelu",
        )
        return TransformerLM(cfg)

    if model_name == "transformer_swiglu_matched":
        cfg = TransformerConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=seq_len,
            ffn_type="swiglu_matched",
        )
        return TransformerLM(cfg)

    if model_name == "recurrent_ffn":
        cfg = RecurrentFFNConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            recurrent_state_size=state_dim,
            dropout=dropout,
            max_seq_len=seq_len,
        )
        return RecurrentFFNLM(cfg)

    if model_name == "recurrent_ffn_hybrid":
        cfg = RecurrentFFNConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            recurrent_state_size=state_dim,
            dropout=dropout,
            max_seq_len=seq_len,
            variant="hybrid",
        )
        return RecurrentFFNLM(cfg)

    if model_name == "recurrent_ffn_readout":
        cfg = RecurrentFFNConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            recurrent_state_size=state_dim,
            dropout=dropout,
            max_seq_len=seq_len,
            variant="readout",
        )
        return RecurrentFFNLM(cfg)

    recurrent_variants = {
        "recurrent_ffn_matrix": ("matrix", 0.0),
        "recurrent_ffn_multiscale": ("multiscale", 0.0),
        "recurrent_ffn_predictive": ("predictive", 0.0),
        "recurrent_ffn_stable": ("stable", 0.0),
        "recurrent_ffn_ultra": ("ultra", 0.0),
        "recurrent_ffn_sparse": ("sparse", 0.0),
        "recurrent_ffn_aux": ("auxiliary", 0.1),
    }
    if model_name in recurrent_variants:
        variant, aux_loss_weight = recurrent_variants[model_name]
        cfg = RecurrentFFNConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            recurrent_state_size=state_dim,
            dropout=dropout,
            max_seq_len=seq_len,
            variant=variant,
            aux_loss_weight=aux_loss_weight,
        )
        return RecurrentFFNLM(cfg)

    if model_name.startswith("triple-"):
        cfg = triple_latent_config(
            model_name,
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            state_dim=state_dim,
            dropout=dropout,
            max_seq_len=seq_len,
        )
        return TripleLatentLM(cfg)

    cfg = variant_config(
        model_name,
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
        dropout=dropout,
        deep_embed_dim=max(d_model // 2, 32),
        max_seq_len=seq_len,
        pad_token_id=pad_token_id,
    )
    return UniMatrixLM(cfg)


def evaluate_lm(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in iter_eval_batches(tokens, seq_len, batch_size, device, max_batches=max_batches):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "bpb": mean_loss / math.log(2.0),
        "perplexity": math.exp(mean_loss),
    }


def _checkpoint_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def save_language_model_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    state_dim: int,
    dropout: float,
    pad_token_id: int,
    vocab_size: int,
    seed: int,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "model_config": {
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "state_dim": state_dim,
            "dropout": dropout,
            "pad_token_id": pad_token_id,
        },
        "seed": seed,
        "state_dict": _checkpoint_state_dict(model),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_language_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, Dict[str, object]]:
    device = device or default_device()
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    config = payload["model_config"]
    model = build_model(
        model_name=payload["model_name"],
        vocab_size=config["vocab_size"],
        seq_len=config["seq_len"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        state_dim=config["state_dim"],
        dropout=config["dropout"],
        pad_token_id=config["pad_token_id"],
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def train_language_model(
    model_name: str,
    dataset_name: str,
    output_dir: str | Path,
    steps: int = 150,
    batch_size: int = 16,
    seq_len: int = 128,
    eval_batches: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    state_dim: int = 32,
    dropout: float = 0.1,
    grad_clip: float = 1.0,
    seed: int = 7,
    save_checkpoint: bool = True,
    device: torch.device | None = None,
    lr_warmup_steps: int = 0,
) -> Dict[str, object]:
    device = device or default_device()
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    corpus = load_byte_corpus(dataset_name, cache_dir=Path(output_dir) / "cache")
    expected_corpus_name = {
        "wikitext2": "wikitext2-bytes",
        "tinyshakespeare": "tinyshakespeare-bytes",
    }[dataset_name]
    if corpus.name != expected_corpus_name:
        raise RuntimeError(
            f"Requested dataset '{dataset_name}' but loaded '{corpus.name}'. "
            "Aborting to avoid mixing benchmark results across corpora."
        )
    model = build_model(
        model_name=model_name,
        vocab_size=corpus.vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        state_dim=state_dim,
        dropout=dropout,
        pad_token_id=corpus.pad_token_id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if lr_warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step_index: min(float(step_index + 1) / float(lr_warmup_steps), 1.0),
        )
    train_history: List[float] = []
    wall_start = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        x, y = random_lm_batch(corpus.train_tokens, seq_len, batch_size, device, generator)
        optimizer.zero_grad(set_to_none=True)
        logits, aux_loss = model(x)
        lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss = lm_loss
        aux_weight = getattr(model, "get_aux_loss_weight", lambda: 0.0)()
        if aux_loss is not None and aux_weight > 0.0:
            loss = loss + aux_weight * aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        train_history.append(lm_loss.item())

        if step % max(steps // 5, 1) == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[LM] {model_name} step {step:04d}/{steps} lr={current_lr:.6f} loss={lm_loss.item():.4f}")

    duration = time.perf_counter() - wall_start
    metrics = evaluate_lm(model, corpus.valid_tokens, seq_len, batch_size, device, max_batches=eval_batches)

    result = {
        "task": "language_modeling",
        "model": model_name,
        "dataset": corpus.name,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "seed": seed,
        "device": str(device),
        "train_seconds": duration,
        "train_loss_last": train_history[-1],
        "train_loss_mean_last_10": sum(train_history[-10:]) / min(len(train_history), 10),
        "val_loss": metrics["loss"],
        "val_bpb": metrics["bpb"],
        "val_perplexity": metrics["perplexity"],
        "num_params": int(model.get_num_params()),
        "lr_warmup_steps": lr_warmup_steps,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_checkpoint:
        checkpoint_path = save_language_model_checkpoint(
            path=out_dir / f"{model_name}_lm.pt",
            model=model,
            model_name=model_name,
            dataset_name=dataset_name,
            seq_len=seq_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            state_dim=state_dim,
            dropout=dropout,
            pad_token_id=corpus.pad_token_id,
            vocab_size=corpus.vocab_size,
            seed=seed,
        )
        result["checkpoint_path"] = str(checkpoint_path)
    out_file = out_dir / f"{model_name}_lm.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small byte-level language model.")
    parser.add_argument("--model", required=True, choices=GENERIC_LM_MODELS)
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "tinyshakespeare"])
    parser.add_argument("--output-dir", default="auto_research_llm_ideas/results/lm")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--no-save-checkpoint", action="store_true")
    args = parser.parse_args()

    result = train_language_model(
        model_name=args.model,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_batches=args.eval_batches,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        state_dim=args.state_dim,
        dropout=args.dropout,
        seed=args.seed,
        save_checkpoint=not args.no_save_checkpoint,
        lr_warmup_steps=args.lr_warmup_steps,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
