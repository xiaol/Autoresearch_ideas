# UniMatrix-Structured (Low-Rank + Diagonal State)

![UniMatrix-Structured](../assets/arch/umt_structured.png)

**Mechanism.** Factorize the matrix state inside the Universal Transformer loop into a low-rank component plus a diagonal component.

**Why it might help.** Increases effective state capacity while keeping memory and compute closer to linear scale.

**Tradeoffs.** Approximation error can hurt quality; tuning rank vs diagonal balance is critical.
