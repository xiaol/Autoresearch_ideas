# Generic Latent Follow-Up

This note answers the natural objection to the corrected higher-order benchmark:
if `typed-latent` is benchmark-aware, does latent compression still help on a
generic benchmark with no typed parsing or symbolic heads?

## What we ran

We evaluated the untyped recurrent latent family already present in the repo:

- `transformer`: standard attention baseline
- `triple-latent`: recurrent pair-memory latent
- `triple-slot`: slot-compressed pair-memory latent
- `triple-hybrid`: pair-memory latent plus local convolution

All models were run on the same generic suite:

- byte-level WikiText-2 LM
- associative recall
- sequence-length throughput

Two regimes are useful:

1. Matched width: all models use `d_model=64`, `n_layers=3`, `n_heads=4`, `state_dim=16`
2. Approximate parameter match for LM only:
   - `transformer`: `d_model=64` (`174,848` params)
   - `triple-latent`: `d_model=60` (`170,424` params)
   - `triple-slot`: `d_model=60` (`170,520` params)
   - `triple-hybrid`: `d_model=56` (`169,424` params)

## Command

```bash
python -m auto_research_llm_ideas.experiments.run_generic_latent_followup \
  --output-root auto_research_llm_ideas/results/generic_latent_followup \
  --lm-steps 80 \
  --memory-steps 200 \
  --bench-seq-lens 64 128 256 512 \
  --d-model 64 \
  --n-layers 3 \
  --n-heads 4 \
  --state-dim 16
```

This writes:

- `matched_width/lm_vs_standard_attention.csv`
- `matched_width/memory_vs_standard_attention.csv`
- `matched_width/benchmark_vs_standard_attention.csv`
- `param_matched_lm/lm_vs_standard_attention.csv`

Measured pilot outputs from the current run live in:

- `results/_generic_latent_v1/`
- `results/_generic_latent_parammatched_v1/`
- `results/_triple_latent_budget_sweep_v1/`
- `results/_winner_vs_transformer_seeds_v1/`
- `results/_winner_triple_hybrid_d64s8/`

## Measured result

Matched-width LM:

- `transformer`: `5.124` BPB, `174,848` params
- `triple-latent`: `5.022` BPB, `187,928` params
- `triple-slot`: `5.041` BPB, `188,024` params
- `triple-hybrid`: `4.766` BPB, `208,472` params

Approximate parameter-matched LM:

- `triple-latent`: `5.114` BPB at `170,424` params
- `triple-slot`: `5.126` BPB at `170,520` params
- `triple-hybrid`: `5.067` BPB at `169,424` params

Near-budget sweep winner:

- `triple-hybrid` with `d_model=64`, `state_dim=8`: `4.827` BPB at `177,752` params
- the same winner averages `4.806` BPB across three seeds
- the corresponding Transformer baseline averages `5.143` BPB across the same three seeds

Associative recall:

- `transformer`: `25.4%`
- all three generic latent variants: `13.4%`

Throughput:

- absolute throughput is still much worse than attention in this reference implementation
- `triple-hybrid` is about `29.7x` slower than the Transformer at sequence length `64`
- `triple-hybrid` is about `14.9x` slower at sequence length `512`
- the latent variants are much flatter across length: their tokens/sec is nearly unchanged from `64` to `512`
- the near-budget winner remains much slower too: about `16.0k` tok/s at length `64` versus `472.6k` for the Transformer

## Interpretation

The important positive result is that the generic latent idea is not limited to
the benchmark-aware `typed-latent` construction. Even without typed parsing or
task-specific heads, the recurrent triple-latent family improves WikiText-2 in
this small pilot regime, and the `triple-hybrid` variant still beats the
Transformer after approximate parameter matching. A targeted near-budget sweep
strengthens the claim further: the best `triple-hybrid` setting beats the
Transformer across all three tested seeds, not just in one lucky run.

The equally important caveat is that this is still not a compute win:

- training is much slower in the current Python-loop implementation
- throughput is far below the optimized attention baseline
- the models still fail on the explicit retrieval-heavy memory task

So the honest claim is:

- latent compression can help on a generic sequence benchmark
- latent compression can realize true triple-token interactions on the corrected synthetic benchmark
- neither result is yet enough to claim a practical replacement for standard attention
