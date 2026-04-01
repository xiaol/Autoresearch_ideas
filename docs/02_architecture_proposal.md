# Architecture Proposal: UniMatrix-ROSA

This proposal uses a **Universal Transformer** (shared parameters across depth steps) with a **matrix-valued recurrent state**, plus RWKV-inspired DeepEmbed and ROSA for long-range routing. The goal is a fast, memory-stable, attention-free backbone with strong long-context generalization. [M2RNN] [RWKVv8]

## Design Goals
- Maintain matrix-valued recurrent state for linear-time scaling. [M2RNN] [EagleFinch]
- Use Universal Transformer depth recurrence to share parameters and improve data efficiency.
- Improve long-range signal routing without attention via ROSA-style suffix structure. [RWKVv8]
- Increase token-conditioned expressivity via DeepEmbed-style channel modulation. [RWKVv8]
- Preserve hardware efficiency via state expansion and blockwise training. [M2RNN]
- Enable spectral stability and hybrid update rules for better long-context control.

## High-Level Block
Each **UT step** combines three streams:
1. **Matrix-State Recurrence:** A non-linear matrix-valued update that replaces attention.
2. **ROSA Context:** A suffix-automaton memory signal blended into the residual stream.
3. **DeepEmbed Modulation:** A high-dimensional token embedding that modulates channels inside the FFN.

### Notation (Per Layer)
Let `x_t` be the token hidden state at time `t`.

Matrix-state update (per head):
```
k_t, v_t, q_t = W_k x_t, W_v x_t, W_q x_t
u_t = phi(k_t), w_t = phi(v_t), z_t = phi(q_t)
S_t = (1 - g_t) ⊙ S_{t-1} + g_t ⊙ (u_t w_t^T)
y_t = S_t z_t
```
Where `S_t` is a matrix-valued state (per head), `phi(·)` is a nonlinearity, and `g_t` is a learned gate. [M2RNN] [EagleFinch]

ROSA fusion:
```
r_t = Rosa(x_{≤t})
x'_t = x_t + W_r r_t
```
ROSA provides lossless long-range information propagation without attention. [RWKVv8]

DeepEmbed modulation:
```
d_t = DeepEmbed(token_t)
ffn(x'_t) = W_2 σ(W_1 x'_t ⊙ (1 + W_d d_t))
```
Token-specific high-dimensional embeddings modulate FFN channels. [RWKVv8]

## Model Name (Working)
**UniMatrix-ROSA** (Universal Transformer with matrix-state recurrence + ROSA + DeepEmbed).

## Expected Advantages
- **Linear-time** recurrence without KV cache growth. [M2RNN] [GatedDeltaNet]
- **Long-range generalization** by combining matrix-state recurrence with ROSA. [M2RNN] [RWKVv8]
- **Higher token expressivity** from DeepEmbed, especially for rare tokens. [RWKVv8]

## Implementation Notes
- Share parameters across depth steps (Universal Transformer).
- Use multi-head matrix state (shape: `B × H × D × D`).
- Add a state expansion factor `E` to increase tensor-core utilization, then project back. [M2RNN]
- For training efficiency, consider blockwise or scan-based parallelization of recurrence.
- ROSA is wired in `model/rosa_memory.py` as a placeholder; replace with a true suffix automaton later.
- Add optional spectral-radius regularization or eigenvalue clamping to stabilize very long contexts.

## Variants
Standalone architectures for ablations live in `docs/architectures/` with PNGs in `assets/arch/`.
