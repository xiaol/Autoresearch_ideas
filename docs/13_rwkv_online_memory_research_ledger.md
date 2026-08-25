# RWKV-MS Online Memory: Experiment Ledger and Next Ideas

## Snapshot

This document summarizes the experimental campaign in
[`xiaol/Multi-state-RWKV-online-memory`](https://github.com/xiaol/Multi-state-RWKV-online-memory)
through source commit
`3857849898c309d1ede869497eba4b8bb3516b3c` (2026-08-25), pulled on
2026-08-26. It records what worked, what failed, what each result rules out,
and which ideas remain worth testing.

The central conclusion is:

> Multi-state allocation and RWKV-7 state updates work well in controlled
> recall, and projected-slot online memory improves a native task. In the
> frozen-Gemma hybrid, however, no tested route has yet shown that the
> row-correct RWKV recurrent state is the cause of a causal or generation gain.

This distinction matters. A memory path can be nonzero, trainable, sensitive to
layer placement, and even improve a benchmark while still failing to prefer the
correct source over a matched donor.

## Claim Boundary

### Established

- Dynamic state allocation beats fixed blocking on the controlled DLA recall
  task at matched state count.
- With identical imperfect boundaries, the RWKV-7 state update is more robust
  than the linear block-sum state in the tested synthetic regimes.
- A surprise-selected exact cache can complement a compressed RWKV state; a
  recency cache does not recover far needles in the same HOLA-style test.
- A frozen-Gemma projected-key/value slot system passed its locked publisher
  validation. Scene-boundary micro-F1 rose from `0.1820` to `0.2727`, a
  `+0.0907` gain.
- Open-tensor cumulative depth routing can identify the correct source. Its
  four anchor top-1 rates were `0.9375 / 0.8750 / 0.9375 / 1.0000`.

### Not established

- The validated projected-slot result does not establish RWKV recurrence gain;
  its active `projected_kv_slots` readout bypasses the RWKV matrix scan.
- No recurrent candidate has passed the full correct-versus-zero,
  correct-versus-matched-donor, and correct-versus-layer-permuted causal gate.
- Teacher-forced identity at answer positions has not transferred reliably to
  the causal predictor state or autoregressive generation.
- Full-Bandwidth-style top-to-bottom feedback has not been authorized because
  it would renew computation on a value path that is still donor-ambiguous.
- No recurrent-memory SOTA claim is supported by this campaign.

## What Was Tried

| Family | Core idea | Strongest evidence | Decision |
| --- | --- | --- | --- |
| DLA and multi-state RWKV | Allocate recurrent states at adaptive boundaries | Perfect recall in several synthetic settings; RWKV-7 remains stronger under noisy or low-capacity boundaries | Keep as the mechanism baseline |
| HOLA-style hippocampus | Add a surprise-selected exact cache beside compressed state | Weakest low-`K` case improved from `0.669` to `0.880`; matched recency cache stayed at `0.665` | Keep as a capacity branch |
| Projected-slot outer memory | Use content-addressed projected K/V slots around frozen Gemma | Passed publisher validation; scene micro-F1 `+0.0907` | Valid system result, not recurrent evidence |
| Recurrent carrier-controller | Let RWKV state modulate the projected carrier with an exact projected-only zero path | Beat a fresh projected-only control by mean `+0.00730` scene F1, but correct state lost to zero and donor controls | Recurrent cause rejected |
| Recurrent-only and addressed value | Make RWKV state the material value; use no projected value bypass | Structural and gradient screens passed; native margins stayed below the locked `+0.005` gate | Retired |
| Chunk-aligned addressing | Align one projected key with each touched RWKV chunk | Improved internal layer margins, but donor and layer-permuted generation beat correct recurrence | Retired |
| Native RWKV routing | Remove projected addressing and route over recurrent slots directly | State interventions were material, but causal/native specificity did not emerge | Retired |
| Output gates, MoE, and FFN modulation | Use scalar/vector gates, addressed MoE, sparse outer FFNs, or DeepEmbed channel modulation | Repeatedly proved state presence and layer sensitivity; matched donor was neutral or better | Retire one-pass gate-shape tuning |
| Learned write perturbations | Inject projected address information into RWKV `k/v/a/b` or learned low-rank writes | Nonzero trainable effects and layer separation; donor margin remained approximately zero | Retired in tested form |
| Direct identity losses | Contrast projected keys or projected values with correct and donor RWKV reads | Training audits passed; held-out donor preference did not | Retired targets; key and value geometries were mismatched |
| Bilinear and Full-Bandwidth-inspired bridges | Pair query and state through compatibility gates or a mandatory full-vector CrossGLU | Observer/mechanics screens could pass; causal donor gates failed | Retired as identity repairs |
| Algebraic address binding | Rotary, diagonal-sign, bidirectional-sign, or discrete PLMSC codes | Rotary did not commute with channelwise updates; sign and PLMSC routes missed exactness/collision gates | Retired |
| Shadow identity probes | Learn source identity from frozen state features | Exact-v5 answer-position shadow reached `0.954545` held-out donor row accuracy | Diagnostic pass only |
| Causal shadow and prompt transport | Recover identity before answer tokens or latch it from the prompt boundary | Causal token gate failed; prompt latch preserved `0.9375` selection but did not make the value useful | Transport partly solved; value use failed |
| Continuous query-aligned writes | Map write address into RWKV's native right-axis `k/a/b` geometry, leaving `v` unchanged | Retrieval and mechanics passed; causal zero margin was `+0.050419`, but donor margin was `-0.007690` | Retired readout family |
| Address-decoded reconstruction | Decode `S d(A)` into a token/value representation | Mean reconstruction cosine `0.912683`; donor separation only `0.008718`, with `0/42` modules passing | Retired linear decoder |
| Virtual K/V | Use address-derived virtual keys and RWKV contractions as virtual values | Linear and co-rotated keys failed held-out selection; local compatibility passed only `2/4` anchors | Retired address-only selector maps |
| Cumulative virtual routing | Accumulate compatibility evidence through depth before selecting a virtual value | Open routing identity passed all anchors | Keep the selector, retire the carrier |
| Live virtual suffix | Append routed RWKV values as ephemeral attention suffix entries | Identity remained strong, but zero/provider-off and slot-permutation invariants failed | Retired exact suffix carrier |
| Source-canonical residual renewal | Select a source cumulatively, then inject a bounded RWKV residual at an earlier layer | Target selection reached `0.9375`; mean CE could improve, but donor-positive rows stayed below threshold | Useful diagnostic, not promoted |
| Source-bound outer FFN | Train a small mandatory state-valued FFN after source selection | Correct memory improved CE by `0.230696`; donor-positive rows were only `0.59375` | Retired |
| Divergent-token supervision | Move contrast from the shared JSON opener to the first target/donor-divergent answer token | Improved the divergent token, but source selection fell to `0.40625` there | Exposed answer-phase selector drift |
| Prompt-latched joint identity | Freeze prompt-boundary source identity across answer tokens and gate one selected read | Selection stayed at `0.9375`; donor and layer-roll often beat correct state | Retired terminal-read family |
| Prompt-latched multi-anchor bundle | Concatenate native reads from layers `5/11/17` as a mandatory 96-D value | Latest held-out run failed: donor margin `-0.01847`, correct gain `-0.03022`, donor-positive rows `0.25` | Retired frozen-query bandwidth route |

## Lessons That Generalize

### 1. State presence is not state identity

Correct-versus-zero is necessary but weak. Many candidates made zero state much
worse while a matched donor was equal or better. This means the model learned
to depend on a recurrent feature without learning which episode supplied it.

Every future experiment should treat these as separate questions:

1. Is the path active?
2. Is the correct layer/source selected?
3. Does the selected value improve the next causal prediction?
4. Does that improvement survive autoregressive generation?

### 2. Addressing and value transport are different bottlenecks

The cumulative router showed that source selection can be strong while the
selected RWKV read remains causally unhelpful. Conversely, a high-bandwidth
value path cannot repair a donor-neutral selector. New designs should expose a
separate selector score and material value path, then intervene on each one
independently.

### 3. Teacher forcing hides a temporal boundary

Source identity was recoverable at answer positions and at the prompt boundary,
but degraded at the causal predictor and later divergent answer tokens. A
prompt latch fixes source transport, yet does not guarantee that the latched
state contains the answer-specific value needed by the decoder.

### 4. More depth cannot manufacture identity

Earlier residual injection and multi-anchor bundling made state effects larger,
but did not reliably make correct state better than donor state. Full-Bandwidth
feedback should remain gated until a one-pass donor-specific causal value path
exists; otherwise it only amplifies an ambiguous memory.

### 5. Exact counterfactuals are more informative than aggregate gains

The campaign's strongest methodological contribution is its intervention set:
projected-only/provider-off, zero state, matched donor, layer permutation,
address-only swap, state-only swap, joint swap, shuffle, and cache/state
immutability. These controls prevented a projected-memory improvement from
being mislabeled as recurrent-memory success.

### 6. Negative results should retire mechanisms, not merely settings

When a family passes mechanics but repeatedly fails matched-donor causal use,
increasing gain, batch size, duration, or gate capacity does not target the
failure. The next experiment should change the information path or supervision
boundary.

## Ranked Next Experiments

### P0 — Learn the associative query inside the selected matrices

The latest frozen-query multi-anchor bundle rules out simple bandwidth as the
main bottleneck. The next materially distinct test should change the query used
to read the already-selected matrices:

```text
source*, confidence* = latch(cumulative_prompt_router)
q_l' = r_l + U_l V_l h_t                 # low-rank learned query delta
m_l  = normalize(S_l[source*] q_l')       # l in {5, 11, 17}
bundle = concat(m_5, m_11, m_17)
residual = W_out(U_bundle(bundle) * sigmoid(G_h(h_t)))
```

Constraints:

- hidden state may form the associative query and gate, but cannot bypass the
  RWKV value path;
- all state-side maps remain bias-free, so zero matrices are exactly
  provider-off;
- freeze the cumulative selector and prompt latch to isolate query learning;
- train and evaluate on source- and pair-disjoint open data first;
- require both the first-token and first-divergent-token views to pass before
  protected mechanics or causal access.

This is the direct follow-up already implied by the failed multi-anchor
ablation: learn where to read inside the chosen state instead of adding another
gate after an unhelpful terminal contraction.

### P1 — Pretrain recurrent value retrieval before LM fusion

If P0 still selects the right source but fails donor-specific CE, the writer and
state may not preserve a decodable source-specific value. Test that directly in
a small standalone task before involving frozen Gemma:

- write multiple `(source, key, value)` episodes into the exact RWKV recurrence;
- query a source/key after distractors and require exact value recovery;
- compare correct, matched-donor, wrong-key, layer-roll, and zero controls;
- vary state count and boundary policy independently;
- promote the writer/read pair only after held-out source and key
  generalization pass.

This separates a representation failure from an LM-integration failure and
avoids spending four-GPU runs on a state that never encoded the needed value.

### P2 — Factor identity, capacity, and age

Run a small factorial mechanism study with one frozen selector/readout:

| Axis | Levels |
| --- | --- |
| Source identity | correct, donor, layer-permuted |
| State capacity | one slot, current multi-slot bank, historical anchor bank |
| Age | recent, medium, far |
| Auxiliary exact cache | off, surprise-selected, recency-selected |

Use the MARCH-inspired historical anchors and the successful HOLA-style
surprise cache here. This reveals whether failures come from overwrite/age or
from source/value identity. Do not mix this capacity branch into P0 until the
base associative query passes donor-specific controls.

### P3 — Add calibrated abstention only after identity passes

A confidence controller may fall back to projected-only output when recurrent
evidence is weak. It is useful only after the recurrent path prefers correct
state to donor state. Before that point, abstention can hide identity failure by
learning generic state-presence or task-policy shortcuts.

### P4 — Test Full-Bandwidth temporal feedback last

Only after a one-pass recurrent value path passes mechanics and causal donor
gates should the selected value re-enter a shallow layer for another pass.
Snapshot memory so feedback never double-writes it, use the proposed
`75% / 22% / 3%` one/two/three-pass mixture with prefix mixing, and require the
last feedback deltas to contract without CE divergence. Re-run all donor,
permutation, zero, shuffle, and disabled controls before native evaluation.

## Autoresearch Gate Sequence

Use the following promotion ladder for every new family:

1. **Mechanism:** exact equations, finite outputs, mutation where expected,
   zero-state identity, and cache/projected-carrier immutability.
2. **Open identity:** correct source must beat matched donor and layer
   permutation on source- and component-disjoint rows.
3. **Open causal:** correct state must lower causal CE, not merely change
   logits; report mean margin and positive-row fraction.
4. **Protected causal:** open exactly once under a signed protocol only after
   the open gates pass.
5. **Native generation:** compare correct, zero/projected-only, donor, and
   layer-permuted states under the same checkpoint and decoder.
6. **Claim:** report the narrowest mechanism supported by the interventions.

Minimum controls for all recurrent claims:

| Control | Question answered |
| --- | --- |
| Provider off / projected-only | Does recurrence add anything beyond the carrier? |
| Zero recurrent state | Is the recurrent value path active? |
| Matched donor state | Is source-specific content used? |
| Layer-permuted state | Is layer placement causally meaningful? |
| Address-only and state-only swaps | Is selection separate from value transport? |
| Joint source permutation | Is the implementation invariant to candidate ordering? |
| Cache/state byte hashes | Did the intervention mutate unrelated state? |

## Stop List

Do not spend the next run on:

- another scalar/vector gate or gain sweep over the same terminal read;
- more updates on the retired addressed-value, DeepEmbed, CrossGLU, PLMSC,
  sign-binding, continuous-write-readout, or virtual-suffix families;
- a larger address-only virtual-key mapper;
- Full-Bandwidth feedback before donor-specific causal value passes;
- native benchmark access justified only by mechanics, routing, or
  teacher-forced identity.

The shortest path forward is P0: preserve the cumulative source selector and
prompt latch, learn a low-rank query inside each selected RWKV matrix, and make
the resulting multi-anchor bundle the only material output path.
