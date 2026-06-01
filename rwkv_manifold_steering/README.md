# RWKV Manifold Steering Reproduction

This folder packages the RWKV/Qwen reproduction of Goodfire-style manifold steering for cyclic concepts. It is meant to be runnable as an isolated Python project inside this repository.

The experiment follows the idea from:

- Goodfire article: <https://www.goodfire.ai/research/the-world-inside-neural-networks>
- Paper: *Steering Along Manifolds to Control Neural Networks*, arXiv:2605.05115
- Original codebase used as reference: <https://github.com/goodfire-ai/causalab/tree/manifold_steering>

## What This Adds

The original method is transformer-centered. This reproduction adapts the intervention target to RWKV-7 by patching the last-token block output after the selected RWKV block's residual updates. That is the closest analogue we used to transformer residual-stream block output patching.

The saved report runs use matched endpoints:

- manifold steering and linear steering start from the same hidden state,
- both end at the same hidden state,
- only the intermediate path differs.

This removes endpoint mismatch as a behavior-space confound.

## Contents

- `src/rwkv_manifold_steering/`: runnable experiment code.
- `scripts/`: convenience scripts for smoke tests, Qwen runs, comparisons, and video narration.
- `reports/manifold_report/metrics.json`: reproduced matched-endpoint metrics.
- `reports/manifold_report/visuals/*/summary.json`: per-run summaries.
- `reports/manifold_report/visuals/*/*.png`: selected plots from the reproduced runs.
- `reports/manifold_report/audience_video/`: narration source, MiniMax render report, and contact sheet.
- `docs/`: method notes, reproduction guide, and audience Q&A.

Large generated MP4/GIF/HTML artifacts are not included here by default. They can be regenerated from the scripts and saved report outputs.

## Setup

From this folder:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The Qwen experiments download models from Hugging Face. If a gated model is used, set `HF_TOKEN` or run `huggingface-cli login`.

For the runs in the report, the intended small models are:

- RWKV: a local RWKV-7 0.1B `.pth` checkpoint. Set `RWKV_MODEL_PATH=/path/to/model.pth`, or place it at `models/rwkv7-0.1b.pth`.
- Qwen: the small Qwen model configured in the scripts/code path used by `qwen-weekday-manifold`.

## Quick Smoke Test

```bash
bash scripts/run_smoke.sh
```

This verifies the package imports and runs a reduced path without reproducing the full report.

## Reproduce Matched Report Runs

The main generalized runner is `rwkv_manifold_steering.cyclic_manifold`.

Run all four matched report cases:

```bash
bash scripts/run_matched_report.sh
```

Example RWKV weekday run:

```bash
python -m rwkv_manifold_steering.cyclic_manifold \
  --backend rwkv \
  --task weekday \
  --out-dir outputs/report_weekday_rwkv_matched \
  --start Monday \
  --end Thursday \
  --linear-endpoint-mode matched
```

Example Qwen weekday run:

```bash
python -m rwkv_manifold_steering.cyclic_manifold \
  --backend qwen \
  --task weekday \
  --out-dir outputs/report_weekday_qwen_matched \
  --start Monday \
  --end Thursday \
  --linear-endpoint-mode matched
```

Example month runs:

```bash
python -m rwkv_manifold_steering.cyclic_manifold \
  --backend rwkv \
  --task month \
  --out-dir outputs/report_month_rwkv_matched \
  --start January \
  --end April \
  --linear-endpoint-mode matched

python -m rwkv_manifold_steering.cyclic_manifold \
  --backend qwen \
  --task month \
  --out-dir outputs/report_month_qwen_matched \
  --start January \
  --end April \
  --linear-endpoint-mode matched
```

If command-line defaults drift, inspect:

```bash
python -m rwkv_manifold_steering.cyclic_manifold --help
```

## Saved Results

Matched-endpoint summary:

| task | model | layer | acc | r manifold | r linear | dist manifold | dist linear |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weekday | RWKV-7 0.1B | 11 | 0.143 | 0.949 | 0.891 | 0.0128 | 0.0133 |
| weekday | Qwen3.5 0.8B | 2 | 0.143 | 0.857 | 0.537 | 0.5203 | 0.5228 |
| month | RWKV-7 0.1B | 5 | 0.083 | 0.989 | 0.210 | 1.9228 | 1.9216 |
| month | Qwen3.5 0.8B | 20 | 0.083 | 0.915 | 0.744 | 0.0638 | 0.0639 |

The tiny models are at chance-level task accuracy. Treat this as a geometry/intervention sanity check, not a claim that the models solve weekday or month arithmetic.

## Dimensionality Reduction

The activation embedding dimensionality is reduced with PCA from `sklearn.decomposition.PCA`.

There are two uses:

- Experiment manifold fitting: PCA to `--pca-dim`, default `16`, then standardization and periodic cubic spline fitting over the cyclic concept angle.
- Visualization: PCA to `3` dimensions for interactive/static 3D plots.

PCA is a linear projection. We did not use UMAP, t-SNE, an autoencoder, or a learned nonlinear reducer for the reported runs.

See [docs/audience_questions.md](docs/audience_questions.md) for the short audience-facing answer.

## Regenerate Narrated Video

The narrated video uses MiniMax TTS. Credentials are expected outside the repo, for example in `~/.codex/env/PaperX.env`, with:

```bash
MINIMAX_API_KEY=...
MINIMAX_GROUP_ID=...
```

Set `PAPERX_ENV=/path/to/env` if your credential file lives elsewhere.

Then:

```bash
python -m rwkv_manifold_steering.audience_video --repo . --profile narrated
python scripts/generate_audience_minimax_audio.py
```

The script strips `A:` and `B:` speaker labels before TTS. Speaker labels are only used to select voices.
