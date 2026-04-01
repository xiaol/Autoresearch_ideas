from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Dict, List

import torch

from auto_research_llm_ideas.experiments.data import BYTE_PAD, BYTE_VOCAB_SIZE, default_device, set_seed
from auto_research_llm_ideas.experiments.train_lm import GENERIC_LM_MODELS, build_model


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def current_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 ** 2)
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def benchmark_models(
    models: List[str],
    output_dir: str | Path,
    seq_lens: List[int],
    batch_size: int = 8,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    state_dim: int = 32,
    dropout: float = 0.1,
    warmup_iters: int = 5,
    timed_iters: int = 20,
    seed: int = 11,
    device: torch.device | None = None,
) -> List[Dict[str, object]]:
    device = device or default_device()
    set_seed(seed)
    results: List[Dict[str, object]] = []

    for model_name in models:
        max_seq_len = max(seq_lens)
        model = build_model(
            model_name=model_name,
            vocab_size=BYTE_VOCAB_SIZE,
            seq_len=max_seq_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            state_dim=state_dim,
            dropout=dropout,
            pad_token_id=BYTE_PAD,
        ).to(device)
        model.eval()

        for seq_len in seq_lens:
            tokens = torch.randint(0, BYTE_VOCAB_SIZE, (batch_size, seq_len), device=device)

            for _ in range(warmup_iters):
                with torch.no_grad():
                    model(tokens)
            synchronize(device)

            start_memory = current_memory_mb(device)
            start = time.perf_counter()
            for _ in range(timed_iters):
                with torch.no_grad():
                    model(tokens)
            synchronize(device)
            elapsed = time.perf_counter() - start
            end_memory = current_memory_mb(device)

            tokens_processed = batch_size * seq_len * timed_iters
            results.append(
                {
                    "task": "throughput_benchmark",
                    "model": model_name,
                    "device": str(device),
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "timed_iters": timed_iters,
                    "num_params": int(model.get_num_params()),
                    "tokens_per_second": tokens_processed / max(elapsed, 1e-9),
                    "milliseconds_per_iter": (elapsed / timed_iters) * 1000.0,
                    "memory_mb": max(start_memory, end_memory),
                }
            )
            print(
                f"[BENCH] {model_name} seq={seq_len:4d} "
                f"tok/s={results[-1]['tokens_per_second']:.1f} mem={results[-1]['memory_mb']:.1f}MB"
            )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark UniMatrix throughput scaling.")
    parser.add_argument("--models", nargs="+", default=GENERIC_LM_MODELS[:4])
    parser.add_argument("--output-dir", default="auto_research_llm_ideas/results/benchmark")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--timed-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    results = benchmark_models(
        models=args.models,
        output_dir=args.output_dir,
        seq_lens=args.seq_lens,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        state_dim=args.state_dim,
        dropout=args.dropout,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
        seed=args.seed,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
