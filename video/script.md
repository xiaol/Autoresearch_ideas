# Video Script (Draft)

**Hook**
We keep hearing that attention is the only path to scale, but matrix-state RNNs are quietly catching up. M2RNN shows that non-linear, matrix-valued recurrence can scale and even outperform strong linear-attention baselines in hybrid form. [M2RNN]

**Context**
RWKV already uses matrix-valued states and dynamic recurrence to get linear-time scaling. That makes it a natural bridge between classic RNNs and modern LLM training. [EagleFinch]

**New Ingredients**
RWKV v8 adds two ideas that are perfect for a next-gen matrix-RNN: DeepEmbed for token-conditioned channel modulation, and ROSA, a suffix-automaton memory that can propagate information without attention. [RWKVv8]

**Proposed Model**
We combine matrix-state recurrence with ROSA memory signals and DeepEmbed modulation inside a Universal Transformer loop. This is the **UniMatrix-ROSA** architecture: linear-time, stable memory growth, and stronger long-range routing than plain recurrence. [M2RNN] [RWKVv8]

**Why It Might Win**
M2RNN already reports strong scaling and hybrid gains versus Gated DeltaNet. If we can keep the matrix-state core but add lossless long-range propagation and richer token modulation, we might reach transformer-level quality without attention. [M2RNN] [GatedDeltaNet] [RWKVv8]

**What We’ll Test**
We’ll run ablations on DeepEmbed and ROSA, evaluate long-context state tracking, and measure speed/throughput at scale. Success is matching hybrid perplexity with a fully recurrent model. [M2RNN]

**Close**
Matrix-state RNNs are no longer a curiosity; they’re a viable path to scalable language modeling. If this works, attention becomes optional rather than mandatory.
