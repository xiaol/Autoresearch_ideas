# UniMatrix-SparsePointer (Sparse Slot Cache)

![UniMatrix-SparsePointer](../assets/arch/umt_sparsepointer.png)

**Mechanism.** Keep the shared recurrent UniMatrix core, but add a sparse external slot cache with a learned write gate, top-k lookup, and pointer-style value fusion back into the readout.

**Why it might help.** The current recall failure looks like overwrite and dilution: the compressed recurrent state has to absorb keys, values, and filler tokens into one matrix. SparsePointer instead stores only salient events and retrieves them explicitly when the query arrives.

**Tradeoffs.** Needs a write policy, eviction rule, and slot-allocation strategy. If slot routing collapses or writes become too dense, the memory can degenerate back into a noisy cache.
