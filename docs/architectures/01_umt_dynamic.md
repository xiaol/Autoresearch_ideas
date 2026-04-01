# UniMatrix-Dynamic (Per-Head Timescale)

![UniMatrix-Dynamic](../assets/arch/umt_dynamic.png)

**Mechanism.** Add per-head timescale gates inside the Universal Transformer recurrence to dynamically control decay per head.

**Why it might help.** Different heads can specialize to different memory horizons, reducing interference between short- and long-term signals.

**Tradeoffs.** Timescale collapse is possible without regularization; adds small gating overhead.
