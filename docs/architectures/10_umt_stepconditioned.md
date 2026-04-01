# UniMatrix-StepConditioned (UT Step Gates)

![UniMatrix-StepConditioned](../assets/arch/umt_stepcond.png)

**Mechanism.** Condition the update rule on the Universal Transformer step index using step embeddings and step gates.

**Why it might help.** Enables different behavior across recurrent depth: early steps can focus on local refinement while later steps emphasize long-range consolidation.

**Tradeoffs.** Needs careful scheduling; poorly tuned step gates can reduce parameter sharing benefits.
