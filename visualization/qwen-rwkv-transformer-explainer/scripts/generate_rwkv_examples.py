#!/usr/bin/env python3
"""Generate static RWKV-7 traces for the example dropdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwkv_trace import DEFAULT_MODEL_PATH, build_trace, load_rwkv


EXAMPLES = [
    "The sun rises in the",
    "Machine learning models learn from training",
    "A language model predicts the next",
    "Attention helps tokens focus on relevant",
    "A small language model can answer simple",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default="static/rwkv-traces")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sampling-type", choices=["top-k", "top-p"], default="top-k")
    parser.add_argument("--sampling-value", type=float, default=5)
    parser.add_argument("--selection-strategy", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_rwkv(args.model_path)

    summary = []
    for index, prompt in enumerate(EXAMPLES):
        trace = build_trace(
            loaded=loaded,
            input_text=prompt,
            temperature=args.temperature,
            sampling_type=args.sampling_type,
            sampling_value=args.sampling_value,
            selection_strategy=args.selection_strategy,
            top_n=args.top_n,
            seed=args.seed,
        )
        path = output_dir / f"example-{index}.json"
        path.write_text(json.dumps(trace, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        summary.append(
            {
                "index": index,
                "prompt": prompt,
                "sampled": trace["sampled"],
                "modelId": trace["modelId"],
                "modelMeta": trace["modelMeta"],
            }
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
