# Attention Pretrain Ablation — DSA / LSA / CSA / HCA

Tiny byte-level pretraining ablation that swaps ONLY the causal-attention rule:

| variant | mechanism | per-query budget (final pos, seq 256) |
|---|---|---|
| `dsa` | **oracle** top-k over exact scores (upper bound for a DeepSeek-style learned indexer) | 96 oracle tokens |
| `lsa` | local window 64 + mean-pooled 16-block recall (top-4 blocks → top-32 tokens) + mask reuse every other layer | 64 + 32 = 96 tokens |
| `csa` | local window 64 + gated 4:1 compressed KV, top-32 blocks | 64 + 32 slots = 96 |
| `hca` | local window 64 + gated 8:1 compressed KV, all completed blocks | 64 + 32 slots = 96 |

Budgets are **matched at ~96 slots** by default (`--dsa-topk 96`, `--hca-ratio 8`); every
sparsity knob is a CLI flag. All variants are dense O(t²) simulations with boolean masks —
`tokens_per_sec` measures masking overhead, **not** sparse-kernel efficiency.

## Files
- `train_attention_variants.py` — model + trainer (fixed; see Audit below)
- `check_attention_correctness.py` — 68-check empirical suite: exact-zero gradient causality
  (t ∈ {33,77,101,128}), degenerate-limit equivalence to dense attention, LSA mask confinement,
  odd-length block padding, minimal-dataset regression, degenerate-config rejection
- `build_mix_from_parquet.py` — corpus builder that HTTP range-reads HF parquet shards directly
  (footer + row groups only) when `datasets` streaming is unavailable; reuses the canonical
  source specs/processing from `prepare_rwkv_ms_hf_dataset.py`
- `compare_runs.py` — multi-seed comparison table + val-loss curves + `comparison.json`

## Audit findings fixed (2026-07-02)
1. **Latent uniform-softmax leak** — an all-masked score row (`topk=0`/`window=0`) softmaxes to
   *uniform*, silently attending to the future. Degenerate configs now rejected in
   `ModelConfig.__post_init__`, including the `ratio > local_window+1` coverage blind zone.
2. **Budget confound** — budgets were 64/96/96/72; any ranking confounded mechanism with
   allowance. Defaults now matched at 96.
3. **`get_batch` off-by-one** — final window unreachable; minimal dataset falsely rejected.
4. **`metrics.jsonl` append contamination** — same `--run-name` reruns interleaved rows; now truncated.
5. **`tokens_per_sec`** included eval wall time; now train-only (`train_sec` also logged).

## Result (budget-matched, 2026-07-03)
Corpus: `rwkv_ms_hf_mix_50mchars` (50.0M chars, 7 sources; built via parquet range-reads).
Model: 6L / 8H / 256d ≈ 4.93M params, seq 256, bf16. One pass ≈ 49.2M tokens/run, seeds {1337, 2024}.

| variant | final val (mean, nats) | cross-seed spread |
|---|---|---|
| lsa | **1.3132** | 0.0126 |
| dsa | 1.3222 | 0.0025 |
| csa | 1.3344 | 0.0042 |
| hca | 1.3435 | 0.0032 |

Both seeds produce the identical ordering `lsa < dsa < csa < hca`. lsa-vs-dsa (0.0089) sits
inside lsa's seed spread — too close to call. Token-level selection (lsa/dsa) vs compressed
memory (csa/hca) is a repeatable gap (≥0.012 nats, ~3× spread): **at equal budget, keeping real
tokens beats keeping blurred summaries** — at this toy scale.

Artifacts: `/run/media/xiaol/B214449214445C0B/attention_pretrain_ablation/budget96_6k_seed{1337,2024}/`
and `budget96_comparison/`. Explainer video project: `~/X/attention_ablation_manim/`.

## Run
```bash
.venv=/home/xiaol/X/HRM-Text/.venv/bin/python  # torch 2.11 cu130
$venv train_attention_variants.py \
  --data-dir /run/media/xiaol/B214449214445C0B/autoresearch_datasets/rwkv_ms_hf_mix_50mchars \
  --run-name my_run --steps 6000 --batch-size 32 --n-layer 6 --n-head 8 --n-embd 256
$venv check_attention_correctness.py   # must print ALL CHECKS PASSED
```
