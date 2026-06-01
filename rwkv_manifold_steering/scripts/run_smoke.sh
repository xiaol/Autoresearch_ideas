#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${RWKV_MODEL_PATH:-models/rwkv7-0.1b.pth}"

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_manifold_steering.weekday_manifold \
  --model "$MODEL_PATH" \
  --out-dir outputs/weekday_manifold \
  --num-steps 50 \
  --n-prompts 16 \
  --start Monday \
  --end Thursday
