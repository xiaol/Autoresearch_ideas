# UniMatrix-ROSA (Suffix Memory)

![UniMatrix-ROSA](../assets/arch/umt_rosa.png)

**Mechanism.** Inject a suffix-automaton memory vector into the Universal Transformer residual stream alongside matrix-state recurrence.

**Why it might help.** ROSA provides lossless long-range routing that complements recurrence, improving state tracking over very long contexts.

**Tradeoffs.** Implementation is non-trivial and may require custom memory structures or CPU offload.
