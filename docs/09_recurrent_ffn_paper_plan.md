# Recurrent FFN Paper Plan

This note narrows the current research direction to a single benchmark-first question:

**Can we replace the Transformer FFN/MLP with a modern recurrent memory block while keeping attention fixed?**

That framing is intentionally narrower than "replace Transformers with RNNs." It isolates the role of the FFN and gives us a cleaner paper story.

## Why This Direction

The strongest conceptual bridge is:

- Transformer FFNs appear to function partly as **static memories** stored in weights.
- Modern recurrent and state-space models such as RWKV, Mamba, xLSTM, and M$^2$RNN provide **dynamic token-conditioned memory**.
- A recurrent FFN replacement therefore tests whether static FFN memory can be upgraded to dynamic recurrent memory without changing self-attention.

This is the right question if we want a paper that is:

- more novel than a plain activation swap
- cleaner than a full RWKV or Mamba backbone replacement
- easier to benchmark honestly against existing baselines in this repo

## Main Claim To Test

Replace the dense FFN sublayer in a decoder block with a compact recurrent memory module:

```text
x -> attention -> residual -> recurrent FFN -> residual
```

The attention path stays unchanged. Only the FFN slot changes.

This gives us a precise hypothesis:

> A recurrent FFN can improve memory-sensitive and state-tracking behavior, and may match or improve language-model quality at similar active parameter or compute budget.

## Architecture Scope

The first paper should stay conservative.

### What stays fixed

- decoder-only causal LM setup
- self-attention
- residual structure
- normalization stack
- tokenizer and training recipe for comparisons

### What changes

- the FFN/MLP block after attention

### First candidate module

Use a **Selective Recurrent FFN** with:

- a compact recurrent state per layer
- signed forget or transition gates
- input-conditioned write gate
- output projection back to model width

Minimal form:

```text
u_t = norm(h_t)
f_t = tanh(W_f u_t)
i_t = sigmoid(W_i u_t)
v_t = W_v u_t
q_t = W_q u_t
s_t = f_t ⊙ s_{t-1} + i_t ⊙ v_t
y_t = W_o phi(q_t ⊙ s_t)
```

This is deliberately simpler than a full Mamba-style or RWKV-style backbone. If it works, later variants can add matrix-valued state, low-rank transition, or shared-depth recurrence.

## Paper Positioning

The novelty claim is **not**:

- RNNs beat Transformers
- Mamba is better than attention
- FFNs are unnecessary

The novelty claim **is**:

- Transformer FFNs can be reinterpreted as static memories
- a localized recurrent memory block is a viable FFN replacement
- the benefits and failures can be measured independently of attention changes

That is the claim reviewers can evaluate cleanly.

## Benchmark-First Development Plan

We should not lock the title or abstract before the benchmark pattern is clear.

### Phase 1: mechanism validation

Use the existing synthetic recurrent-memory harness in [`/Users/xiaol/x/PaperX/lld_paper`](/Users/xiaol/x/PaperX/lld_paper):

- associative recall
- signal-vs-noise interference
- throughput

This phase answers:

- does the recurrent FFN train at all
- does it actually improve memory-sensitive behavior
- is the runtime penalty tolerable

### Phase 2: small LM validation

Use the pilot LM path in [`/Users/xiaol/x/PaperX/auto_research_llm_ideas`](/Users/xiaol/x/PaperX/auto_research_llm_ideas):

- byte-level WikiText-2
- memory stress task
- RULER core subset

This phase answers:

- is the model still a plausible language model
- do gains transfer beyond synthetic memory
- does long-context degradation improve or worsen

### Phase 3: external evaluation

If the pilot signal is real, then move to:

- `lm-evaluation-harness`
- broader long-context evaluation
- stronger recurrent baselines at larger scale

This phase determines whether the architecture is a paper about:

- general LLM quality
- memory or state tracking
- efficiency
- or a negative but still useful benchmark result

## Concrete Baseline Set

Use the smallest baseline set that still makes the paper credible.

### Core baselines

- standard Transformer FFN baseline
- `mamba2`
- `gated_deltanet`

These are already wired into [`/Users/xiaol/x/PaperX/lld_paper/experiments/models.py`](/Users/xiaol/x/PaperX/lld_paper/experiments/models.py).

### Secondary baselines

- the strongest `triple-hybrid` or related latent recurrent variant already in this repo
- any FFN-shape or shared-depth internal variant that emerges during ablation

RWKV should be discussed in the paper framing, but it does not need to be the first local benchmark baseline unless we actually implement a clean FFN-local RWKV variant.

## Benchmark Commands To Reuse

### 1. Recurrent-memory smoke test

```bash
PYTHONPATH=/Users/xiaol/x/PaperX/lm-engine:/Users/xiaol/x/PaperX \
python -m lld_paper.experiments.run_benchmarks \
  --models lld gated_deltanet mamba2 softmax_attention \
  --tasks associative_recall signal_noise_interference \
  --steps 50 \
  --eval-batches 8 \
  --seq-len 64 \
  --batch-size 8 \
  --timed-iters 3 \
  --output-dir lld_paper/results/smoke
```

Adapt this runner to include the recurrent FFN model once the implementation exists.

### 2. Small LM and memory suite

```bash
python -m auto_research_llm_ideas.experiments.run_full_suite \
  --output-root auto_research_llm_ideas/results/recurrent_ffn_pilot \
  --models transformer unimatrix-core unimatrix-rosa unimatrix-discovery \
  --lm-steps 80 \
  --memory-steps 200 \
  --bench-seq-lens 64 128 256 512 \
  --d-model 64 \
  --n-layers 3 \
  --n-heads 4 \
  --state-dim 16 \
  --ruler-tasks niah_single_1 vt cwe fwe \
  --ruler-samples 8 \
  --ruler-max-seq-length 128
```

This command already gives the right shape of pilot evidence:

- LM quality
- memory accuracy
- throughput
- RULER core subset

### 3. Generic latent follow-up

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

This is useful if the recurrent FFN story becomes more about latent compression or parameter reuse than about explicit sequence-state gains.

## Preliminary Result: April 1, 2026

We now have a first nontrivial synthetic benchmark run for the initial `recurrent_ffn` baseline.

### Run details

- harness: [`/Users/xiaol/x/PaperX/lld_paper/experiments/run_benchmarks.py`](/Users/xiaol/x/PaperX/lld_paper/experiments/run_benchmarks.py)
- output: [`/Users/xiaol/x/PaperX/lld_paper/results/recurrent_ffn_pilot_v1/summary.json`](/Users/xiaol/x/PaperX/lld_paper/results/recurrent_ffn_pilot_v1/summary.json)
- device: `mps`
- models: `recurrent_ffn`, `mamba2`, `softmax_attention`
- settings: `120` steps, `batch_size=8`, `seq_len=64`, `eval_batches=64`, `hidden_size=64`, `num_layers=2`, `num_heads=4`

### Measured task results

| Task | recurrent_ffn | mamba2 | softmax_attention |
|---|---:|---:|---:|
| associative recall accuracy | 15.82% | 10.16% | 7.03% |
| associative recall eval loss | 2.7375 | 2.7388 | 2.7675 |
| signal-noise accuracy | 12.70% | 8.59% | 8.59% |
| signal-noise eval loss | 2.7426 | 2.7309 | 2.7432 |

### Measured throughput snapshot

| Seq len | recurrent_ffn | mamba2 | softmax_attention |
|---|---:|---:|---:|
| 64 tok/s | 26.2k | 37.8k | 609.8k |
| 128 tok/s | 23.4k | 39.7k | 931.5k |
| 256 tok/s | 22.9k | 42.9k | 511.5k |

### Honest interpretation

- The first `recurrent_ffn` baseline already wins the two synthetic memory tasks on **accuracy**.
- On associative recall it is also the best model on **eval loss**, although only slightly ahead of `mamba2`.
- On signal-vs-noise interference it wins on **accuracy** but not on **eval loss**, where `mamba2` is slightly better.
- The current implementation is clearly **not** a systems win yet. It is slower than `mamba2` and dramatically slower than the optimized softmax attention baseline.

This means the current paper story is **not** "better general architecture" yet.

The current paper story **is**:

- localized recurrent FFN replacement improves small-scale memory-sensitive behavior
- the mechanism is promising
- the runtime story is still weak in the present reference implementation

## Preliminary LM Result: April 1, 2026

We also ran a first matched-parameter byte-level WikiText-2 pilot for the localized `recurrent_ffn` idea.

### Run details

- harness: [`/Users/xiaol/x/PaperX/auto_research_llm_ideas/experiments/train_lm.py`](/Users/xiaol/x/PaperX/auto_research_llm_ideas/experiments/train_lm.py)
- output: [`/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_lm_pilot_v1/summary.json`](/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_lm_pilot_v1/summary.json)
- device: `mps`
- models: `transformer`, `recurrent_ffn`
- settings: `80` steps, `batch_size=16`, `seq_len=128`, `d_model=64`, `n_layers=3`, `n_heads=4`, `state_dim=128`

### Measured LM results

| Model | Params | Val BPB | Val perplexity | Train seconds |
|---|---:|---:|---:|---:|
| transformer | 174,848 | 5.144 | 35.35 | 1.65 |
| recurrent_ffn | 173,888 | 5.178 | 36.21 | 11.37 |

### Measured throughput snapshot

| Seq len | transformer | recurrent_ffn |
|---|---:|---:|
| 64 tok/s | 457.9k | 21.4k |
| 128 tok/s | 481.6k | 26.2k |
| 256 tok/s | 392.7k | 27.4k |

### Honest interpretation

- On the first matched-parameter LM pilot, `recurrent_ffn` is **slightly worse** than the Transformer baseline on WikiText-2.
- The model is still far slower in the current implementation.
- Combined with the synthetic-memory results, the present evidence supports a **memory-sensitive win** rather than a general LM win.

## Tuned Recurrent-FFN Follow-Up: April 1, 2026

We then tested two quality-oriented variants:

- `recurrent_ffn_readout`: adds token-conditioned state readout `q_t ⊙ s_t`
- `recurrent_ffn_hybrid`: splits the FFN budget between a local gated path and a recurrent readout path

We also tightened the LM harness so it now aborts if the requested dataset does
not match the loaded corpus. This prevents the previous silent fallback from
WikiText-2 to Tiny Shakespeare.

### Result summary file

- output: [`/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_tuned_compare_v1/summary.json`](/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_tuned_compare_v1/summary.json)

### 120-step tuned comparison

Settings:

- dataset: `wikitext2-bytes`
- device: `mps`
- recipe: `lr=4e-4`, `dropout=0.0`, `steps=120`
- width: `d_model=64`, `n_layers=3`, `n_heads=4`, `seq_len=128`

| Model | State dim | Params | Val BPB | Val perplexity | Train seconds |
|---|---:|---:|---:|---:|---:|
| transformer | 128 | 174,848 | 4.309 | 19.82 | 1.94 |
| recurrent_ffn | 128 | 173,888 | 4.512 | 22.81 | 22.08 |
| recurrent_ffn_readout | 96 | 168,032 | 4.473 | 22.21 | 25.70 |
| recurrent_ffn_readout | 128 | 198,848 | 4.418 | 21.38 | 26.15 |
| recurrent_ffn_hybrid | 128 | 174,080 | 4.414 | 21.32 | 38.68 |

Additional hybrid scaling:

| Model | State dim | Params | Val BPB | Val perplexity |
|---|---:|---:|---:|---:|
| recurrent_ffn_hybrid | 192 | 223,328 | 4.460 | 22.01 |
| recurrent_ffn_hybrid | 256 | 272,576 | 4.435 | 21.63 |

Interpretation:

- Better optimization matters a lot for these recurrent FFN variants.
- The simple `recurrent_ffn` baseline improves strongly under tuning, but it is still clearly behind the Transformer.
- The best matched-parameter recurrent variant at 120 steps is `recurrent_ffn_hybrid`.
- Larger hybrid width does **not** help in this small pilot, so the main gain seems to come from architecture plus optimization, not from simply adding parameters.

### 200-step two-seed comparison

We then asked whether the remaining gap was just optimization lag by extending
training to `200` steps with the same tuned recipe.

| Seed | Model | Params | Val BPB | Val perplexity | Train seconds |
|---|---|---:|---:|---:|---:|
| 7 | transformer | 174,848 | 3.930 | 15.24 | 2.69 |
| 7 | recurrent_ffn_hybrid | 174,080 | 3.899 | 14.92 | 56.25 |
| 11 | transformer | 174,848 | 3.965 | 15.62 | 2.64 |
| 11 | recurrent_ffn_hybrid | 174,080 | 3.876 | 14.69 | 56.02 |

### Throughput reminder

The quality win does **not** translate into a systems win in the current
reference implementation:

| Seq len | transformer tok/s | recurrent_ffn_hybrid tok/s |
|---|---:|---:|
| 64 | 380.3k | 14.0k |
| 128 | 447.4k | 14.4k |
| 256 | 407.8k | 14.5k |

### Updated interpretation

- The original plain `recurrent_ffn` result was a memory win but not an LM win.
- A stronger localized recurrent replacement with token-local gating and recurrent readout can become a **small-scale LM win** under a tuned recipe and longer optimization.
- The win is currently narrow:
  - small byte-level WikiText-2 pilot
  - two seeds
  - large training-time slowdown
  - no fused recurrent kernel
- The honest paper claim has therefore shifted from
  `memory win only`
  to
  `small-scale quality win plus memory win, but still not a systems win`.

## Result-Dependent Paper Framing

We should choose the paper title only after the metrics settle.

### Case A: broad win on LM plus memory

Use a direct architecture claim.

Possible titles:

- **Dynamic Memory FFNs for Decoder-Only Transformers**
- **Replacing Transformer FFNs with Recurrent Memory**
- **Selective Recurrent Feed-Forward Networks**

Use this framing if:

- LM loss or BPB improves
- memory tasks improve
- long-context degradation is competitive

### Case B: memory win, mixed LM result

Use a narrower capability claim.

Possible titles:

- **Recurrent FFN Replacements Improve State Tracking in Decoder-Only Transformers**
- **From Static FFNs to Dynamic Recurrent Memories**
- **Recurrent Memory Layers as FFN Replacements**

Use this framing if:

- synthetic memory tasks clearly improve
- LM quality is near-parity or slightly mixed
- long-context behavior is at least not catastrophic

### Case C: efficiency or parameter win

Use a systems-aware claim.

Possible titles:

- **Shared Recurrent FFNs for Efficient Transformer Language Models**
- **Parameter-Efficient Recurrent FFN Replacements**

Use this framing if:

- quality is roughly tied
- active parameters or memory improve
- throughput or decoding behavior improves in a meaningful regime

### Case D: negative or mixed result

Still publishable if the analysis is sharp.

Possible titles:

- **What Do Transformer FFNs Still Do Better Than Recurrent Memory Blocks?**
- **A Benchmark Study of Recurrent FFN Replacements in Decoder LMs**
- **Static vs Dynamic Memory in Transformer FFNs**

Use this framing if:

- recurrent FFNs help one narrow axis
- but fail to transfer broadly
- and the ablations explain why

This is not failure. It is still a useful paper if the diagnosis is honest and mechanistic.

### Current best framing after the first pilot

Based on the April 1, 2026 pilot, the most defensible framing is currently **Case B**.

Best title candidates right now:

- **Recurrent FFN Replacements Improve State Tracking in Decoder-Only Transformers**
- **From Static FFNs to Dynamic Recurrent Memories**
- **Recurrent Memory Layers as FFN Replacements**

I would avoid a broad title about efficiency or general language-model superiority until the LM and long-context runs are in.

## Reviewer-Facing Risks

These are the main reviewer objections we should expect:

- "This is just RWKV or Mamba in disguise."
- "You changed sequence mixing, not just FFN behavior."
- "The gains are only synthetic."
- "The runtime is worse than the dense FFN baseline."

The first paper should answer them directly:

- keep attention unchanged
- localize recurrence to the FFN slot
- report both synthetic and LM results
- include throughput and memory, not just accuracy

## Success Criteria

The strongest believable success pattern is:

- recurrent FFN beats the Transformer FFN on memory-sensitive tasks
- recurrent FFN is competitive on LM quality
- recurrent FFN is interpretable as a dynamic memory mechanism
- recurrent FFN has a plausible efficiency or parameter story

The first paper does **not** need to prove universal superiority over all Transformers.

## Immediate Next Steps

1. Implement a first recurrent FFN module as a drop-in FFN replacement.
2. Add it to the smallest existing benchmark harness before trying a larger paper rewrite.
3. Run smoke synthetic tasks first.
4. Run the first real synthetic comparison sweep.
5. Run the pilot LM plus memory suite next.
6. Rename the model and paper only after seeing which axis actually wins.

## Immediate Next Experiment

The next highest-value run is **small LM validation**.

Recommended target:

- compare `recurrent_ffn` against the Transformer baseline on byte-level WikiText-2
- keep the architecture change localized to the FFN slot
- report both LM quality and throughput

Recommended ablation:

- sweep `recurrent_state_size` at a small scale, because the current quality-speed tradeoff is likely very sensitive to that dimension

## Recommended Default Name For Now

Use a neutral working name until the benchmarks decide the final story:

**Recurrent FFN** or **Selective Recurrent FFN**

That keeps the implementation and results files stable while the paper title remains flexible.
