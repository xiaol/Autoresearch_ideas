#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/2] Delivery-Time Prediction"
python3 "$ROOT/examples/delivery-time-prediction/scripts/run_baseline.py"
echo
echo "[2/2] Knowledge Search Retrieval"
python3 "$ROOT/examples/knowledge-search-retrieval/scripts/run_retrieval_eval.py"
