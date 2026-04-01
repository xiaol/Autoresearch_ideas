# Lookup Architecture Blueprint

This note turns the recent recall diagnosis into three concrete retrieval variants for the repo. The key observation is simple: the current UniMatrix core is a **compressed state machine**, while associative recall is a **content-addressable lookup** task. The architecture family therefore needs an explicit retrieval path rather than only richer recurrent updates.

## 1. UniMatrix-SparsePointer

**Role.** Minimal, highest-priority implementation.

**State update.**

\[
S_t = \rho_t \odot S_{t-1} + (1 - \rho_t) \odot U_t
\]

**Sparse write.**

\[
g_t = \sigma(w_g^\top x_t), \qquad
i_t^\star = \arg\max_i \mathrm{score}(W_k x_t, k_i)
\]

If \(g_t\) is high enough, write the token into slot \(i_t^\star\) or into the least-recently-used free slot:

\[
k_{i_t^\star} \leftarrow W_k x_t,\qquad
v_{i_t^\star} \leftarrow W_v x_t
\]

**Lookup.**

\[
\beta_{t,i} = \frac{q_t^\top k_i}{\sqrt{d_k}} + b_{\text{age}}(i), \qquad
c_t = \sum_{i \in \mathrm{TopK}(\beta_t)} \mathrm{softmax}(\beta_t)_i \, v_i
\]

**Why this is first.**

- It directly targets the observed failure mode: filler tokens should not overwrite exact bindings.
- It is easy to ablate against the current `unimatrix-assoc`.
- It can support exact or near-exact token recall via pointer-style output fusion.

## 2. UniMatrix-ProductKey

**Role.** Strongest scalability candidate.

Split the query into two halves:

\[
q_t = [q_t^{(a)}; q_t^{(b)}]
\]

Score two factorized codebooks:

\[
\alpha_i = \langle q_t^{(a)}, K_i^{(a)} \rangle,\qquad
\gamma_j = \langle q_t^{(b)}, K_j^{(b)} \rangle
\]

Retrieve from top bucket pairs \((i,j)\):

\[
c_t = \sum_{(i,j)\in \mathcal{T}} \mathrm{softmax}(\alpha_i + \gamma_j)\, V_{ij}
\]

**Why it matters.**

- Large effective memory without dense scan.
- Better long-context story than a flat append-only cache.
- Strong candidate for matched-FLOP experiments against attention.

**Main risk.**

- Factorized memories can be harder to train than dense top-k lookup.
- Writes need balancing, otherwise a few bucket pairs dominate.

## 3. UniMatrix-Relay

**Role.** Most novel paper candidate.

First-hop retrieval produces anchor context:

\[
m_t^{(1)} = \mathrm{Lookup}(q_t, \mathcal{M})
\]

Then refine the query with the anchor and perform a second lookup:

\[
\tilde{q}_t = f(q_t, m_t^{(1)}), \qquad
m_t^{(2)} = \mathrm{Lookup}(\tilde{q}_t, \mathcal{M}_{\text{relay}})
\]

Final fusion:

\[
c_t = g(q_t, m_t^{(1)}, m_t^{(2)})
\]

**Why it is interesting.**

- Supports disambiguation and compositional retrieval.
- More plausible path toward tasks that need “find anchor, then follow relation”.
- Cleaner narrative for paper novelty than just “bigger cache”.

**Main risk.**

- First-hop errors compound.
- Harder to optimize and explain.

## Recommended build order

1. **UniMatrix-SparsePointer**
2. **UniMatrix-ProductKey**
3. **UniMatrix-Relay**

## Benchmark expectations

- **Associative recall:** SparsePointer should be the first serious win candidate.
- **RULER-style retrieval:** ProductKey is the strongest long-context scaling bet.
- **Compositional lookup / multi-step recall:** Relay is the most novel variant to test.
- **WikiText-2 LM:** any gain is secondary; retrieval modules should not be judged only by BPB on tiny pilot runs.

## Critical ablations

- recurrent state only vs recurrent state + retrieval
- dense append-only cache vs sparse gated cache
- single-hop vs two-hop retrieval
- value-vector fusion vs pointer/logit fusion
- fixed slots vs product-key factorization

## Recommendation

If the goal is to **beat attention on retrieval-heavy tasks**, build `UniMatrix-SparsePointer` first. If the goal is to **tell a stronger scaling story**, build `UniMatrix-ProductKey` next. If the goal is to **claim a more novel architecture contribution**, push `UniMatrix-Relay` after the first two are stable.

## First implementation snapshot

`UniMatrix-SparsePointer` is now implemented as the first retrieval upgrade in the repo. The initial sparse-slot version reached:

- `42.19%` accuracy with dropout `0.1`
- `95.07%` accuracy with dropout `0.0`

We then added **pointer-logit fusion**, letting retrieved slots vote directly on the output vocabulary logits. On the same original small associative-recall pilot setting (`d_model=64`, `n_layers=3`, `state_dim=16`, `steps=200`), the upgraded variant reached:

- `75.63%` accuracy with dropout `0.1`
- `99.17%` accuracy with dropout `0.0`

For reference, the published small Transformer pilot in the same setup was `25.44%`, and the earlier dense `unimatrix-assoc` no-dropout follow-up was `26.46%`. This is strong evidence that sparse slot routing plus direct pointer-style output fusion solves much more of the recall failure than the earlier dense append-only cache.

## SparsePointer ablation sweep

We then ran a focused ablation sweep at the same no-dropout pilot scale, training on `4` key-value pairs and evaluating on `4`, `6`, and `8` pairs.

Primary summary files:

- `auto_research_llm_ideas/results/sparsepointer_ablations_d0/summary.csv`
- `auto_research_llm_ideas/results/sparsepointer_ablations_d0/summary.json`
- targeted check: `auto_research_llm_ideas/results/sparsepointer_ablations_d0/targeted/slots16_nogate.json`

Key results:

- `slots=4, pointer logits on`: `12.84 / 12.50 / 12.40%` on `4 / 6 / 8` pairs
- `slots=8, pointer logits on`: `12.79 / 12.45 / 12.45%`
- `slots=16, pointer logits on`: `24.17 / 17.48 / 13.57%`
- `slots=16, pointer logits on, write gate off`: `26.22 / 18.75 / 14.79%`
- `slots=32, pointer logits off`: `95.65 / 90.28 / 86.62%`
- `slots=32, pointer logits on`: `98.93 / 98.34 / 97.90%`
- `slots=32, pointer logits on, write gate off`: `100 / 100 / 100%`

What this says:

- **Capacity is the first hard threshold.** Even though training uses only `4` true bindings, `4` and `8` slots collapse to chance. This means the slots are not storing only the four clean bindings; they are absorbing many structured and filler transitions, so the usable memory budget must exceed the number of logical key-value pairs.
- **The practical transition happens between `16` and `32` slots.** `16` slots are still badly capacity-limited, while `32` slots are enough to preserve near-exact retrieval even when evaluation is widened to `6` and `8` pairs.
- **Pointer logits improve exactness and robustness, not just headline accuracy.** With `32` slots, removing pointer logits still gives strong performance (`95.65%` on `4` pairs), but degradation under `6` and `8` pairs is much larger. Direct token-level voting makes the model more stable under higher retrieval pressure.
- **The current learned write gate is not helping on this benchmark.** Turning the write gate off slightly improves `16` slots and makes `32` slots perfect. The likely reason is that on this synthetic task, the gate sometimes suppresses useful writes, whereas the slot budget is already large enough to tolerate simple always-write behavior.

Current interpretation:

- The main SparsePointer gain does **not** come from clever recurrence.
- It comes from three things working together: enough slot capacity, sparse top-k retrieval, and exact token-level output routing.
- On this benchmark, write selectivity is less important than having enough clean slots and a direct pointer path into the logits.
