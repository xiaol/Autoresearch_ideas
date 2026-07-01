# Adaptive Multi-State Selective Memory (AMS²)

## Unified Architecture for Write-Gated, Boundary-Aware Recurrence

### Motivation

Three systematic weaknesses appear across all papers in this repo:

1. **Parameter-matched comparisons conflate architecture quality with scale** — flagged by Level 5 on every paper
2. **No matched-compute budgeting** — the recurrent variants are 10-20× slower per token but only marginally better on quality
3. **Memory is either global (UniMatrix) or slot-based (recurrent FFN), never both** — no paper combines selective *what* with adaptive *when*

Meanwhile, the two memory modules we ported (delta-mem selective gating, multi-state boundary detection) solve complementary halves of this problem but have never been combined.

### Hypothesis

> A shared-depth Universal Transformer with **adaptive multi-state selective memory** — where the model jointly learns write gates (what to store) and boundary policies (when to segment) — can match Transformer quality at *lower total compute* by avoiding redundant state updates, even if per-token FLOPs are higher.

### Architecture

```
Input x_t
    │
    ▼
┌─────────────────────────────────────┐
│  Change-Point Detector (CPD)        │  ← Multi-state boundary policy
│  - running variance shift            │     (from model/multi_state_memory.py)
│  - boundary score > τ → new segment  │
└─────────────────────────────────────┘
    │
    ▼  per token / per segment boundary
┌─────────────────────────────────────┐
│  Write Gate                          │  ← Selective memory write
│  - sigmoid(MLP(x_t))                 │     (from model/selective_memory.py)
│  - controls delta-rule update rate   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Matrix-State Update (per head)      │  ← UniMatrix core
│  S_t = ρ⊙S_{t-1} + (1-ρ)⊙(k_t v_tᵀ) │     (from model/unimatrix.py)
│  with write-gate-modulated ρ         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Multi-State Readout                 │
│  - weighted mix of head states       │
│  - boundary-reset on segment breaks  │
└─────────────────────────────────────┘
    │
    ▼
  y_t (to next shared depth step)
```

### Key Innovations

1. **Gated retention**: The write gate from selective memory controls the retention rate `ρ` in UniMatrix's matrix-state update: `ρ = ρ_base * (1 - write_gate)`. When the gate is low (nothing new), state decays faster — saving effective capacity for informative tokens.

2. **Boundary-aware reset**: The change-point detector segments the sequence. At segment boundaries, a configurable fraction of state heads reset, mimicking the multi-state DLA advantage without needing N independent full states.

3. **Shared-depth across segments**: Unlike standard multi-state where each segment trains separate parameters, AMS² keeps the Universal Transformer's shared-depth loop — the *state* segments, not the weights. This keeps parameter count flat while letting the model allocate memory capacity to distinct regions of the input.

### Suggested Experiment (from Level 5 ablation idea)

| Step | What | Measures |
|---|---|---|
| 1 | Baseline: UniMatrix-Core (d=128, L=4) on byte-level WikiText-2 | BPB, tokens/sec, memory |
| 2 | Add write-gated retention (modulate ρ via learned gate) | BPB change, gate sparsity |
| 3 | Add boundary-aware reset (CPD from multi-state memory) | BPB change, boundary entropy |
| 4 | Full AMS² (gated ρ + CPD reset + shared depth) | Full metrics vs Transformer at matched *compute* not just matched parameters |

### Why This Wins Over Existing Papers

| Paper | What it misses | What AMS² adds |
|---|---|---|
| UniMatrix | Flat retention rate, no adaptive capacity allocation | Write-gated ρ per token |
| Recurrent FFN | Single vector state, no segmentation | Multi-state with boundary reset |
| delta-mem | External memory, not integrated into recurrence | Unified gated recurrence |
| Multi-State DLA | Fixed boundary policy for all heads | Head-specific CPD thresholds |

### Risk (from Level 5 methodology findings)

The biggest risk is **throughput regression** — adding CPD + gating increases per-token FLOPs. The counterargument from the multi-state DLA results: at matched *state count*, adaptive boundaries outperform fixed blocking even with simpler state updates. If AMS² can converge in fewer steps (because it allocates capacity where it matters), total training compute could decrease even if per-token FLOPs increase.

### Concrete Next Step

The gated retention modulation is already implemented as a minimal diff to `model/unimatrix.py`:

```python
# In UniMatrixBlock.forward(), the retention computation:
base_retention = sigmoid(retention_proj(x_t))          # original
if use_gated_retention:
    write_gate = sigmoid(write_gate_proj(x_t))
    retention = base_retention * (1 - write_gate)       # AMS²
else:
    retention = base_retention                          # original
```

Activate it via config:

```python
cfg = variant_config("unimatrix-ams2", vocab_size=32000)
# Sets: use_gated_retention=True, use_boundary_reset=True, boundary_window=8
model = UniMatrixLM(cfg)
```

The boundary-aware reset runs a running variance change-point detector before the token loop and resets a portion of state heads at detected segment boundaries.

### Experiments

Run the AMS² variant through the existing benchmark suite:

```bash
python -m auto_research_llm_ideas.experiments.run_full_suite \
  --models transformer unimatrix-core unimatrix-ams2 \
  --lm-steps 80 --memory-steps 200 \
  --bench-seq-lens 64 128 256 512 \
  --d-model 128 --n-layers 4 --n-heads 4 --state-dim 32 \
  --output-root results/ams2_pilot
```

Expected outcomes:
- **If gated retention helps**: AMS² matches or improves on UniMatrix-Core BPB at matched steps, with lower state-update frequency (sparser writes)
- **If boundary reset helps**: AMS² improves on associative recall vs UniMatrix-Core without needing sparse pointer slots
- **Combined**: Best case is improvement on both LM quality and memory tasks simultaneously, which UniMatrix-Core cannot do alone
