#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT/dist"
PDF_OUT="$DIST_DIR/machine-learning-full-course-by-ai.pdf"

echo "[1/3] Building mdBook HTML..."
mdbook build "$ROOT"

mkdir -p "$DIST_DIR"

echo "[2/3] Exporting review PDF..."
pandoc "$ROOT/book/print.html" \
  --from=html \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V colorlinks=true \
  -o "$PDF_OUT"

echo "[3/3] Build complete."
echo "HTML: $ROOT/book/index.html"
echo "PDF:  $PDF_OUT"
