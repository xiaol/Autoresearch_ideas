# Term Ledger — every load-bearing term is defined on screen/voice before use

| Term | Plain definition given in narration/visual | First beat |
|---|---|---|
| byte-level language model | predicts the next byte; vocabulary = 256 byte values + 1 end-of-text marker = 257 | harness |
| validation loss (nats) | average surprise on held-out text; lower = better; differences of ~0.01 called out as near-noise | cold open |
| causal attention | a position may read itself and earlier positions, never later ones | harness |
| sparse attention | each query attends to a selected subset of allowed positions instead of all of them | DSA beat |
| top-k selection | keep the k highest-scoring past positions per query | DSA beat |
| oracle selection | selecting with the exact full score matrix — information a real system can't afford; upper bound | DSA beat |
| local window | the last w tokens, always attended at full resolution (w=64 here) | LSA beat |
| block / block recall | older context grouped into fixed-size chunks; query scores chunk summaries to recall a few | LSA beat |
| cross-layer mask reuse | odd layers reuse the previous layer's selection mask instead of computing their own | LSA beat |
| compressed KV | a learned gate pools each block of keys/values into one summary vector | CSA beat |
| compression ratio | tokens per summary vector (4:1 for CSA, 8:1 for HCA) | CSA beat |
| attention budget / slots | how many score columns a query is allowed at the final position (tokens + block summaries) | budget beat |
| NEG_INF masking | forbidden logits set to -1e9 so softmax drives their weight to ~0 | leak beat |
| uniform-softmax leak | softmax over an all-equal (all-masked) row = uniform weights → reads everything incl. future | leak beat |
| gradient-causality test | d(output_i)/d(input_j) must be exactly 0 for all j>i | audit beat |
| off-by-one (sampler) | exclusive upper bound made the last valid training window unreachable | bugs beat |
| ablation | change exactly one factor, hold everything else fixed, compare | cold open |
| bf16 / AMP | reduced-precision training mode (mentioned only as "same trainer settings"; not load-bearing) | — |
