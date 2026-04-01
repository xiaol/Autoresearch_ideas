# UniMatrix-DeepEmbed (Token Modulation)

![UniMatrix-DeepEmbed](../assets/arch/umt_deepembed.png)

**Mechanism.** Use a high-dimensional token embedding to multiplicatively modulate FFN channels inside the Universal Transformer loop.

**Why it might help.** Improves expressivity for rare tokens and enables token-conditioned routing in the FFN.

**Tradeoffs.** Adds memory and can increase overfitting without strong regularization.
