# Knowledge-Growth Ledger — Attention Pretrain Ablation Explainer

## Part thesis
A tiny 4-variant sparse-attention pretraining ablation looked finished and showed "no difference";
auditing the instrument (causality gradients, latent softmax leak, harness bugs, unmatched
attention budgets) and re-running budget-matched at 12x data is what makes the comparison mean anything.

## Viewer question
When a head-to-head ablation of attention mechanisms shows near-identical losses, what must be
verified about the experiment itself before believing the conclusion?

## Before model (misconception)
"If the same trainer runs four attention variants with the same seed and data, whatever loss gap
appears (or doesn't) reflects the mechanisms."

## Obstacle (concrete failures shown)
1. Budget confound: per-query attention-slot budgets differed (64/96/96/72 at the final position)
   — mechanism and budget varied simultaneously.
2. Latent leak: masks enforced by NEG_INF fill + softmax; an all-masked row yields UNIFORM (not zero)
   attention — silent future leakage for degenerate configs (topk=0, window=0).
3. Harness bugs: get_batch off-by-one (last window unreachable; minimal dataset falsely rejected);
   metrics.jsonl append-mode interleaving across reruns; tok/s denominator included eval wall time.
4. Coverage blind zone when compression ratio > local_window+1 (found by review, guarded now).

## Source evidence (all local, verifiable)
- Code: /home/xiaol/X/attention_pretrain_ablation/train_attention_variants.py (before/after diffs)
- Empirical checks: check_attention_correctness.py — 68 checks incl. exact-zero causality gradients
  at t∈{33,77,101,128}, degenerate-limit equivalence to dense attention, block-padding paths.
- Baseline run: /run/media/.../attention_pretrain_ablation/four_attention_tiny_1k_*/summary.json
  (1000 steps, 4.1M tokens: best_val spread 0.0136 nats across variants).
- Fixed run: budget96_6k_seed{1337,2024}/ (6000 steps x batch 32 = 49.2M tokens, 6L/256d ~4.9M params,
  2 seeds) + comparison.json curves.

## Mechanism (what resolves the obstacle)
- Gradient-causality test: dy_i/dx_j must be exactly 0 for j>i (not approximately).
- Config validation making empty selection rows unrepresentable.
- Budget matching: all four variants pinned to ~96 slots at the final query
  (dsa_topk 64→96, hca_ratio 32→8), sparsity knobs exposed as CLI flags.
- Fixed sampler/metrics/timing so the record is trustworthy.

## Local comparison (what the code is a stand-in for)
- DSA here = ORACLE top-k on exact scores — upper bound on DeepSeek's learned lightning indexer,
  at dense cost. LSA = LongCat-style local+recall with cross-layer mask reuse approximating CLI.
  CSA/HCA = NSA-flavored compressed KV branches. None are kernel-level reproductions; tok/s is
  masking overhead, not sparse-kernel efficiency.

## After model (what the viewer can now do)
Read any ablation table and ask: were budgets matched? is masking provably causal (gradient test)?
can a degenerate config leak silently? does the harness contaminate reruns? — and know the
concrete test for each.

## Boundary (not proven here)
~5M params / 49M byte-tokens is a toy scale; byte-level val loss ≠ downstream ability; oracle
selection ≠ deployable indexer; single-machine dense simulation says nothing about real kernels.
The result ranks mechanisms only at this scale and budget.

## Beat where the viewer learns it
The uniform-softmax leak beat (silent future reading) and the budget-morph beat (64/96/96/72 → 96)
are the designated "mental model flips".
