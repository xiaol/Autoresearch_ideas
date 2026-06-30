# 11 — Multi-State Online Memory

## What It Is

Multi-state online memory extends the single-state recurrent cell of RWKV-7 by
maintaining **independent recurrent states for each adaptively segmented block**
of the input sequence. Instead of a single state vector that is updated at every
timestep, the model learns to detect "change points" in the token stream and
resets a subset of its state heads at those boundaries. This allows the model
to:

- Retain long-range dependencies *within* a coherent segment.
- Avoid state pollution when the topic, format, or distribution shifts.
- Reuse a small number of state heads across many segments via the boundary
  reset mechanism.

The design is inspired by the **Multi-State RWKV** architecture described in
[arXiv:2606.10650](https://arxiv.org/abs/2606.10650) (DLA — Dynamic Linear
Attention) and the [Multi-State-RWKV-online-memory
repo](https://github.com/xiaol/Multi-state-RWKV-online-memory) in this workspace.

## How It Extends Single-State RWKV-7

A standard RWKV-7 layer maintains a single recurrent state vector (or matrix)
per head:

```
state_t = state_{t-1} * w_t + (1 - w_t) * (k_t * v_t)
y_t = r_t * state_t
```

This is an efficient linear recurrence, but it has a weakness: every timestep
contributes to the same state, so information from early tokens can be diluted
by later, unrelated tokens. The multi-state variant addresses this by routing
different subsequences to independent state copies.

**Block boundary detection.** A learned `ChangePointDetector` scores each
timestep based on the change in token variance. When the score exceeds a
threshold, the states for one or more heads are reset. This creates a soft
segmentation: each "state head" independently decides when to forget.

**Independent state heads.** `n_heads` independent `RWKV7Head` modules each
maintain their own state vector. Because each head has its own `w_proj` gate,
different heads learn to specialize — some tracking long-range topical
information (low decay, infrequent reset), others tracking local syntactic
cues (higher decay, frequent reset).

**State mixing.** An optional learned `mix_gate` re-weights the contribution of
each head at every timestep, conditioned on the current input and the
concatenated head states. This is analogous to soft-committee routing.

## Change-Point Detection Policy

The `ChangePointDetector` uses a simple but effective heuristic:

1. Compute a running mean of tokens via cumulative averaging.
2. Measure the variance of tokens relative to that running mean.
3. Compute the frame-wise absolute change in variance — a large change suggests
   a distribution shift.
4. Normalize by the max change across the sequence and pass through a learned
   MLP with a sigmoid output (in [0,1]).

The threshold (default 0.6) controls sensitivity. The detector can also be
bypassed by passing an explicit `boundary_mask`, useful for supervised
segmentation or curriculum learning where boundaries are known a priori.

## Connections to Existing Work in This Repo

- **UniMatrix** (`model/unimatrix.py`): UniMatrix already uses a multi-head
  matrix state with separate `state_dim` per head. Multi-state memory can be
  seen as swapping UniMatrix's fixed per-timestep matrix update for a
  boundary-gated vector update. The two are complementary — one could combine
  UniMatrix's matrix-valued states with the boundary detection mechanism.

- **ROSA Memory** (`model/rosa_memory.py`): The Rapid Online Suffix Automaton
  (ROSA) represents a different approach to online memory — it stores exact
  suffix information. Multi-state memory is lighter (vector states) and more
  flexible (learned boundaries). A hybrid where ROSA provides the boundary
  signal and multi-state heads provide the content representation is a natural
  extension.

- **Selective Memory** (`model/selective_memory.py`): Selective memory uses
  delta-rule writes to a slot-based key-value store. Multi-state memory trades
  the slot-store for compressed vector states, reducing memory footprint at the
  cost of representational capacity.

## Suggested Experiments

1. **Single-state vs. multi-state on needle-in-haystack.** Compare a standard
   RWKV-7 head against the multi-state variant on the needle-in-haystack recall
   task. Hypothesis: multi-state should outperform when the haystack contains
   multiple topical shifts, because the state head that observed the needle is
   not overwritten by later segments.

2. **Fixed vs. adaptive boundaries.** Replace the learned
   `ChangePointDetector` with fixed periodic resets (every K tokens) and
   compare. This isolates the value of the boundary *detection* from the value
   of multiple states.

3. **Scaling n_heads.** Measure perplexity and memory recall as `n_heads`
   increases. Intuitively, more heads should help up to the point where the
   mixing gate overfits.

4. **Multi-state as a drop-in for UniMatrix states.** Replace the matrix state
   in a UniMatrix block with a `MultiStateMemory` and train on language
   modeling. Does the boundary-detection signal improve over UniMatrix's
   implicit per-timestep retention?

5. **Curriculum boundary learning.** Start with frequent resets (high
   threshold) and anneal to less frequent resets over training. This may yield
   better convergence than training with the final threshold from scratch.

## References

- arXiv:2606.10650 — *Dynamic Linear Attention: Multi-State RWKV for Online
  Memory*
- https://github.com/xiaol/Multi-state-RWKV-online-memory
- UniMatrix block in `model/unimatrix.py` (this repo)
- ROSA memory stub in `model/rosa_memory.py` (this repo)
- SelectiveMemory in `model/selective_memory.py` (this repo)
