# UniMatrix-Hybrid (Interleaved Attention)

![UniMatrix-Hybrid](../assets/arch/umt_hybrid.png)

**Mechanism.** Interleave matrix-state blocks with lightweight attention blocks inside the Universal Transformer recurrence.

**Why it might help.** A small amount of attention can recover dependencies that recurrence alone misses while keeping most layers linear-time.

**Tradeoffs.** Adds quadratic cost in attention layers and complicates kernel fusion.
