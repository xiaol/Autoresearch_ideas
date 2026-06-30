# Delta-Mem Integration: Selective Online Memory

## What is Delta-Mem?

[Delta-Mem](https://arxiv.org/abs/2605.12357) (arXiv:2605.12357) introduces a selective online memory mechanism for language models. The core insight is that not all inputs deserve equal write priority to a memory store. Delta-Mem uses a learned gating mechanism to determine *what* to write and *when*, combined with a delta-rule update that interpolates between existing memory content and new inputs rather than overwriting.

Key contributions of the paper:

- **Selective write gating**: A lightweight gate network decides, for each memory slot, how much of the new input to integrate.
- **Delta-rule update**: Memory writes are interpolative (`mem <- mem + gate * (new - mem)`), which stabilizes training and prevents catastrophic forgetting.
- **Top-k sparse read**: At each step, only the top-k most relevant memory slots are attended to, keeping inference efficient.
- **Online decoupled memory**: The memory is maintained separately from the transformer residual stream, acting as an associative store that grows stale entries are gradually updated.

## How Selective Online Memory Works

The `SelectiveMemory` module implements a simplified version of the Delta-Mem idea, adapted for integration into the UniMatrix architecture:

```
Input x_t (d_model)
    |
    +---> write_gate(x_t) --> per-slot write strength (B, N)
    |
    +---> key_proj(x_t)  --> k_t     (slot_dim)
    +---> query_proj(x_t) --> q_t     (slot_dim)
    +---> value_proj(x_t) --> v_t     (slot_dim)
    +---> delta_proj(x_t) --> d_t     (slot_dim)
    |
    Delta-rule updates:
        keys <- keys + gate * (k_t - keys) * 0.1
        vals <- vals + gate * (d_t - vals) * 0.1
    |
    Top-k attention read:
        scores = q_t @ keys^T / sqrt(slot_dim)
        top-k masking, softmax, weighted sum of values
    |
    Output: proj(read) -> d_model
```

**Key design choices:**

1. **Learnable initial slots**: The module starts with learned parameter vectors for slot keys and values, which are expanded per batch. This gives the model a default memory state before any writes occur.

2. **Separate key and delta projections**: The key projection controls which slots are retrieved, while the delta projection controls the *content* written. This decouples routing from storage.

3. **Damped interpolation factor (0.1)**: Rather than learning the interpolation rate separately, we use a fixed small step size. This encourages gradual memory evolution across time steps rather than abrupt overwrites.

4. **Per-timestep loop**: The current implementation processes tokens sequentially in a Python loop (like the existing UniMatrix block does). This is intentional for parity with the existing code structure and could be batched if needed.

## Integration with UniMatrix Architecture

The `SelectiveMemory` module can be wired into `UniMatrixBlock` via a configuration flag `use_selective_memory`. When enabled:

1. The block creates a `SelectiveMemory` instance with `config.slot_dim = config.assoc_dim`, `config.n_slots = config.assoc_slots`, `config.top_k = config.assoc_topk`.
2. After the retention state update and readout, the block optionally queries the selective memory.
3. The memory readout is gated (via `sigmoid(gate)`) and added to the residual stream, mirroring how the existing `assoc_memory` path works.

The proposed `UniMatrixConfig` additions:

```python
use_selective_memory: bool = False
```

The UniMatrix forward loop already processes token-by-token and has `assoc_source` plumbing. The selective memory read would be inserted at the same point as the existing associative memory path, giving the model both its recurrent state update AND a slot-based external memory.

## Differences from Existing RosaMemoryStub

| Aspect | RosaMemoryStub | SelectiveMemory |
|--------|---------------|-----------------|
| Mechanism | Suffix automaton (placeholder stub embeds previous token) | Slot-based key-value memory |
| Write strategy | N/A (stub only) | Learned gated delta-rule |
| Read strategy | Returns prev-token embedding | Top-k attention over slots |
| Capacity | No persistent state (stateless stub) | N learnable slots with evolving content |
| Online state | No (placeholder) | Yes, slots accumulate across time steps |

The `RosaMemoryStub` is a placeholder for a future suffix-automaton (ROSA) implementation inspired by RWKV v8. Delta-Mem is a fundamentally different approach: instead of representing the sequence as a DAG of suffixes, it uses a fixed-size slot array with content-based addressing.

## Suggested Experiments

1. **Selective Memory vs No Memory**: Compare perplexity on standard language modeling benchmarks (WikiText-103, PG-19) between the baseline UniMatrix and UniMatrix with `use_selective_memory=True`. Hypothesis: selective memory helps on longer contexts where the recurrent state decays.

2. **Selective Memory vs Full Attention**: Replace the recurrent state entirely with selective memory reads. This tests whether slot-based memory can substitute for the retention mechanism in UniMatrix.

3. **Associative recall probing**: Use synthetic associative recall tasks (e.g., HINT, multi-query associative recall) where the model must retrieve information seen earlier. Selective memory should outperform no-memory baselines because the slots explicitly store and retrieve key-value associations.

4. **Slot ablation studies**: Vary `n_slots` (e.g., 8, 16, 32, 64) and `top_k` (e.g., 2, 4, 8, 16) to measure the capacity-performance tradeoff. The hypothesis is that performance saturates at modest slot counts, making this efficient for deployment.

5. **Write gate analysis**: Visualize the learned write gate activations. Are certain slots specialized for certain types of content (e.g., noun phrases, numeric values, long-range dependencies)?
