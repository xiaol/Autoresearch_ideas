# UniMatrix-RuleMix (Hybrid Update Rules)

![UniMatrix-RuleMix](../assets/arch/umt_rulemix.png)

**Mechanism.** Learn a token-conditioned mixture over multiple matrix update rules (e.g., outer-product, diagonal, low-rank, and local-conv updates).

**Why it might help.** Lets the model discover which update rule is optimal per token/context instead of hard-coding one rule.

**Tradeoffs.** More parameters and potential training instability if the mixture collapses; may need entropy regularization.
