#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
RWKV_MODEL="${RWKV_MODEL:-}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-0.8B-Base}"

rwkv_model_arg=()
if [[ -n "$RWKV_MODEL" ]]; then
  rwkv_model_arg=(--model "$RWKV_MODEL")
fi

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_manifold_steering.cyclic_manifold \
  --backend rwkv \
  "${rwkv_model_arg[@]}" \
  --task weekday \
  --out-dir outputs/report_weekday_rwkv_matched \
  --start Monday \
  --end Thursday \
  --linear-endpoint-mode matched

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_manifold_steering.cyclic_manifold \
  --backend qwen \
  --model "$QWEN_MODEL" \
  --task weekday \
  --out-dir outputs/report_weekday_qwen_matched \
  --start Monday \
  --end Thursday \
  --linear-endpoint-mode matched

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_manifold_steering.cyclic_manifold \
  --backend rwkv \
  "${rwkv_model_arg[@]}" \
  --task month \
  --out-dir outputs/report_month_rwkv_matched \
  --start January \
  --end April \
  --linear-endpoint-mode matched

PYTHONPATH=src "$PYTHON_BIN" -m rwkv_manifold_steering.cyclic_manifold \
  --backend qwen \
  --model "$QWEN_MODEL" \
  --task month \
  --out-dir outputs/report_month_qwen_matched \
  --start January \
  --end April \
  --linear-endpoint-mode matched

