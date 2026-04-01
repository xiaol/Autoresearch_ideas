# UniMatrix-SkewStable (Skew-Symmetric Update)

![UniMatrix-SkewStable](../assets/arch/umt_skewstable.png)

**Mechanism.** Use a skew-symmetric component in the state update so eigenvalues stay on or near the imaginary axis, improving stability.

**Why it might help.** Controls long-context drift without heavy-handed gating; preserves signal energy across depth steps.

**Tradeoffs.** May reduce expressivity if the skew constraint is too strong; needs a learnable blend with unconstrained updates.
