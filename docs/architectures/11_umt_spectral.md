# UniMatrix-Spectral (Eigenvalue Control)

![UniMatrix-Spectral](../assets/arch/umt_spectral.png)

**Mechanism.** Apply an eigenvalue-aware stability constraint (e.g., spectral radius penalty or eigenvalue clamping) to the matrix update.

**Why it might help.** Directly targets long-context stability by keeping the recurrent operator in a stable regime.

**Tradeoffs.** Adds compute and can overly restrict expressivity if the constraint is too tight.
