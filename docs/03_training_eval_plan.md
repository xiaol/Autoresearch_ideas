# Training and Evaluation Plan

This plan is meant to validate whether **UniMatrix-ROSA** can match or surpass strong linear-attention and RWKV baselines while retaining linear-time inference. [M2RNN] [GatedDeltaNet] [EagleFinch]

## Training Schedule (Proposed)
1. **Sanity Pretrain (Small):** 100M–300M parameters, short context, confirm loss curves and stability.
2. **Mid-Scale:** 1B–3B parameters, longer context, full ablation grid.
3. **Long-Context Stress:** extend context to 8k–64k, measure state-tracking and degradation.

## Key Ablations
- UniMatrix core only vs UniMatrix + DeepEmbed.
- UniMatrix core only vs UniMatrix + ROSA.
- Full UniMatrix-ROSA vs UniMatrix-Hybrid.
- UniMatrix core vs RuleMix / SkewStable / Spectral variants.
- StepConditioned vs non-step-conditioned.
- State expansion factor (E): {1, 2, 4}.
- Gate design: scalar vs per-dimension gating.

## Evaluation Targets
- **Perplexity** on standard LM validation sets.
- **Long-context generalization** tasks (state tracking and long-context benchmarks).
- **Speed**: tokens/sec, memory, and sequence length scaling.
- **Stability**: measure state norm drift and spectral radius over long contexts.

## Long-Context Benchmark Ladder

Use a staged evaluation ladder instead of jumping directly to broad benchmark claims:

1. **RULER core subset** inside this repo:
   - `niah_single_1`
   - `vt`
   - `cwe`
   - `fwe`
2. **Official/full RULER pipeline** once the model is exported into a stronger serving path.
3. **BABILong / NoLiMa / LongBench v2** after the recurrent routing path is no longer a placeholder.

The local `RULER core subset` is useful for rapid model iteration because it is synthetic, self-contained, and exact-match scored. It should still be described honestly as a subset, not as the full official RULER leaderboard setup.

## Success Criteria
- Match or exceed hybrid perplexity while keeping a fully recurrent core. [M2RNN]
- Minimal degradation when scaling from short to long contexts. [M2RNN] [RWKVv8]
- Comparable or faster inference throughput than attention or linear-attention baselines. [M2RNN] [GatedDeltaNet]
