#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${RWKV_MODEL_PATH:-models/rwkv7-0.1b.pth}"

pairs=(
  "Monday Thursday"
  "Tuesday Saturday"
  "Friday Monday"
  "Sunday Wednesday"
)

for pair in "${pairs[@]}"; do
  read -r start end <<<"$pair"
  out_dir="outputs/examples/rwkv_${start}_to_${end}"
  PYTHONPATH=src "$PYTHON_BIN" -m rwkv_manifold_steering.weekday_manifold \
    --model "$MODEL_PATH" \
    --out-dir "$out_dir" \
    --num-steps 40 \
    --n-prompts 12 \
    --start "$start" \
    --end "$end"
done
