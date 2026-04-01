# UniMatrix-DualTimescale (Fast + Slow States)

![UniMatrix-DualTimescale](../assets/arch/umt_dual.png)

**Mechanism.** Maintain fast and slow matrix states with separate gates in the Universal Transformer loop, then mix them into a shared output.

**Why it might help.** Separates short-term pattern extraction from long-horizon tracking, reducing interference.

**Tradeoffs.** More parameters and an extra mixing step; requires careful gate initialization.
