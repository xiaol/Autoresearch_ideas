#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_manifold_steering.compare_models \
  --rwkv outputs/weekday_manifold \
  --qwen outputs/qwen_weekday_manifold \
  --out-dir outputs/model_compare
