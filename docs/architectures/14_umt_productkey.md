# UniMatrix-ProductKey (Factorized Retrieval)

![UniMatrix-ProductKey](../assets/arch/umt_productkey.png)

**Mechanism.** Split the lookup query into two factors, score two smaller codebooks, and retrieve from the top product-key bucket pairs instead of scanning one monolithic memory bank.

**Why it might help.** This gives a much larger effective address space at sublinear lookup cost. It is the strongest scaling candidate if we want explicit retrieval without letting memory cost grow like dense attention.

**Tradeoffs.** Harder to train than plain top-k cache lookup. The codebooks can become imbalanced, and dynamic writes into factorized buckets need careful regularization to avoid hot spots and dead buckets.
