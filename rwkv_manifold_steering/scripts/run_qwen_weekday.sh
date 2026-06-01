#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-0.8B-Base}"

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_manifold_steering.qwen_weekday_manifold \
  --model "$QWEN_MODEL" \
  --out-dir outputs/qwen_weekday_manifold \
  --num-steps 50 \
  --n-prompts 16 \
  --start Monday \
  --end Thursday
