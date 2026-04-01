# Literature Summary (Condensed)

This section captures only the minimum, source-backed facts needed to ground the architecture proposal.

## M2RNN (Matrix-to-Matrix RNN)
M2RNN is a non-linear RNN with matrix-valued hidden states, designed as a scalable alternative to attention for language modeling. The paper emphasizes efficient scaling on GPUs/TPUs via a state expansion method that better utilizes tensor cores, and reports strong scaling plus inference speedups. It also reports that a hybrid M2RNN/attention model outperforms Gated DeltaNet on the Pile by 0.4–0.5 perplexity, and that M2RNN shows perfect state-tracking generalization on longer sequences. [M2RNN]

## RWKV: Matrix-Valued Recurrence and Dynamic Recurrence
The RWKV “Eagle and Finch” work describes RWKV as a linear-time language model with multi-headed matrix-valued states and dynamic recurrence, presenting RWKV as an alternative to attention with strong scaling behavior at length. [EagleFinch]

## RWKV v8 (“Heron”): DeepEmbed and ROSA
The RWKV v8 architecture notes two ideas relevant here: (1) DeepEmbed, which stores high-dimensional token vectors in the FFN for channel-wise modulation and can be offloaded to CPU to reduce GPU memory, and (2) ROSA (Rapid Online Suffix Automaton), described as a neurosymbolic infinite-range lossless information propagator that can replace attention. [RWKVv8]

## Gated DeltaNet (Reference Baseline)
Gated DeltaNet is described as a linear attention architecture that replaces the growing key/value cache with a fixed-size recurrent state. [GatedDeltaNet]

## Immediate Takeaways
- Matrix-valued recurrent state is a recurring motif (M2RNN, RWKV), suggesting it is a strong backbone for scalable RNN LMs. [M2RNN] [EagleFinch]
- M2RNN’s state expansion and hybrid results motivate a design that keeps matrix-state recurrence but adds better long-range propagation (ROSA) and embedding-side modulation (DeepEmbed). [M2RNN] [RWKVv8]
