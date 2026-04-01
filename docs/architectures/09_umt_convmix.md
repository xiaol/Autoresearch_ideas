# UniMatrix-ConvMix (Local + Global Memory)

![UniMatrix-ConvMix](../assets/arch/umt_convmix.png)

**Mechanism.** Blend a local convolutional memory update with the global matrix-state update via a learned mix gate.

**Why it might help.** Improves local pattern capture while preserving long-range memory from the matrix state.

**Tradeoffs.** Adds an extra memory path and can overfit to local patterns if the mix gate saturates.
