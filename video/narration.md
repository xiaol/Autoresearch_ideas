# UniMatrix Architecture Narration

## UniMatrix-Dynamic
We start from the Universal Transformer: the same block is applied across depth steps, h to the next h. UniMatrix replaces attention with a matrix-state update, S_t equals a gated blend of the previous state and a rank-one update. Dynamic adds per-head timescales, so each head chooses its memory horizon.

## UniMatrix-ROSA
UniMatrix-ROSA injects a suffix-memory vector r_t into the residual path. The core still uses y_t equals S_t times q_t, but we add x_t plus W_r r_t for long-range routing. This provides a lossless memory path without attention.

## UniMatrix-DeepEmbed
DeepEmbed modulates the FFN with a token-conditioned vector d_t. The FFN becomes W2 times sigma of W1 x, multiplied by one plus W_d d_t. This makes the update token-aware without extra attention.

## UniMatrix-Structured
Structured state factorizes S_t into L_t R_t transpose plus a diagonal term. That keeps memory large but efficient, while preserving the matrix-state recurrence. It’s a capacity boost without quadratic cost.

## UniMatrix-Hybrid
Hybrid interleaves occasional attention steps with the matrix-state update. Most steps remain linear-time, and attention only appears when needed. It’s a controlled mix: f_attn composed with the matrix step.

## UniMatrix-DualTimescale
DualTimescale maintains fast and slow states with different decay. The output mixes them as alpha S_fast plus one minus alpha S_slow. This separates local patterning from long-range memory.

## UniMatrix-RuleMix
RuleMix learns a mixture over update rules. The update is a sum of alpha_i times delta_i, so the model can discover which rule to apply per token. This is a built-in update-rule search.

## UniMatrix-SkewStable
SkewStable uses a skew-symmetric matrix K_t equals A minus A transpose. The update I plus tau K_t keeps eigenvalues near the imaginary axis, preserving signal energy across steps. It’s stability without heavy gating.

## UniMatrix-ConvMix
ConvMix blends local convolutional memory with the global matrix state. The update is lambda times the matrix delta plus one minus lambda times the conv delta. This captures local and global patterns together.

## UniMatrix-StepConditioned
StepConditioned uses the Universal Transformer step index to gate the update. The gate g(k) controls how much of the new update enters at each depth step. Early steps refine local features; later steps consolidate memory.

## UniMatrix-Spectral
Spectral control constrains the spectral radius of the update, rho of S_t. We can clamp eigenvalues or add a penalty when rho exceeds a target. This is a direct stability guardrail.

## UniMatrix-Discovery
Discovery combines RuleMix, Timescale, ROSA, DeepEmbed, and spectral control. The model can route updates through multiple mechanisms and discover the best mix. It’s the unified search space for architecture discovery.
