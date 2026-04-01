# Open Questions and Risks

1. **ROSA Integration:** How to implement a batch-friendly suffix automaton without breaking throughput? [RWKVv8]
2. **Matrix-State Saturation:** Do non-linear matrix updates accumulate noise at very long contexts? [M2RNN]
3. **State Expansion Costs:** At what expansion factor does extra compute outweigh tensor-core gains? [M2RNN]
4. **DeepEmbed Overhead:** Does token-level modulation increase training instability or memory? [RWKVv8]
5. **Hybrid Necessity:** Is any attention still required to match the best hybrid results reported by M2RNN? [M2RNN]
6. **Spectral Stability:** Which eigenvalue control method best preserves long-context stability without hurting expressivity?
7. **RuleMix Collapse:** Will the mixture over update rules collapse to one rule without explicit regularization?
