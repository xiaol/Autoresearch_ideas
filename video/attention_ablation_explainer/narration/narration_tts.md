# Attention Pretrain Ablation Narration TTS
# FINAL windows: each = measured natural MiniMax duration + 3s intentional hold
# (+2s extra on the closing beat). Written by scripts/apply_windows.py from
# narration/natural_durations.json. Voice plays at exactly 1.0x — never compressed.

## 00:00.00-00:41.05
A: Here is an experiment that looks finished. Four sparse attention mechanisms. A DeepSeek style top-k selector, a local window with long-range recall, and two compressed-memory variants. Trained head to head. Same data, same model, same seed, same budget. After a thousand steps, all four validation losses land within one hundredth of a nat of each other. The obvious reading: at small scale, attention design just does not matter. The correct reading: this race was rigged. Every runner was on a different track, and the stopwatch had three separate defects.

## 00:41.05-01:12.63
A: The testbed is deliberately tiny. A byte-level language model: it predicts the next byte of text, so the vocabulary is just two hundred fifty-seven symbols, every possible byte plus one end-of-text marker. A few transformer layers, a few million parameters, trained on a fixed mix of prose, news, code, and LaTeX. Everything is held constant except one knob: the rule for how each position is allowed to look at the past.

## 01:12.63-01:53.68
A: Variant one borrows DeepSeek's sparse attention idea. Do not attend everywhere; attend where it matters. Each query scores every earlier position, keeps only the top scorers, and runs softmax over that shortlist. But look closely at the implementation. To choose the shortlist, it computes the full score matrix, the exact object a deployable system cannot afford. This is not DeepSeek's cheap learned indexer. It is an oracle. And that changes the question it answers: not "does this mechanism work", but "if sparse selection were perfect, what would perfection be worth?"

## 01:53.68-02:34.19
A: Variant two, in the spirit of LongCat, splits attention in half. A guaranteed local window: the most recent sixty-four tokens, always visible, at full resolution. Plus long-range recall: older context is grouped into blocks of sixteen tokens, each block summarized by its average key. The query scores the summaries, picks its four favorite blocks, and inside them selects thirty-two individual tokens. One more move: every other layer skips building its own selection and reuses the mask from the layer below, a cheap stand-in for cross-layer indexing.

## 02:34.19-03:08.51
A: Variants three and four never un-pool. Recent tokens stay crisp inside the same local window, but the older past exists only as compressed memory: a tiny learned gate decides how much each token contributes to its block's summary key and value. The compressed variant pools four tokens into one and keeps the thirty-two best blocks. The hierarchical variant pools eight into one and keeps them all. Crisp recent tokens and blurred summaries of the past compete inside a single softmax.

## 03:08.51-03:52.26
A: Before trusting any loss number, ask the one question that invalidates everything silently: can position i see the future? Causality is not a vibe. It is a testable property. Take the gradient of output i with respect to every later input. It must be exactly zero. Not small. Zero. A sixty-eight-check suite runs this test across all four mechanisms, at multiple positions, at awkward sequence lengths chosen to stress the block-padding paths, and it cross-checks each mechanism against dense attention in the limit where its sparsity is turned off. Every gradient comes back exactly zero. The masks are honest.

## 03:52.26-04:40.51
A: But the audit found a trap sleeping in the defaults. Every mask here is enforced the same way: forbidden scores are overwritten with minus ten to the ninth, and softmax is trusted to crush them. Now set top-k to zero, or shrink the local window to nothing, and some query rows have every single score at minus ten to the ninth. Softmax of a row where everything is equal is not zero attention. It is uniform attention. The starved query quietly reads every position, including the future, and the validation loss looks wonderful. Nothing crashes. Nothing warns. The fix is boring and absolute: make empty rows unrepresentable. Any configuration that could produce one is rejected at construction.

## 04:40.51-05:12.06
A: Two more instrument errors, both invisible in a plot. The batch sampler had an off-by-one: the final training window of the dataset could never be drawn, and a dataset with exactly one valid window was rejected as too small. And the metrics file was opened in append mode: rerun the same experiment name, and two loss curves interleave into one file, wearing a single config. Neither bug favors any variant. Both corrupt the record you would later trust.

## 05:12.06-05:58.22
A: The deepest problem was not a bug at all. Count what each query may actually touch at the final position. The oracle: sixty-four slots. Local plus recall: sixty-four local plus thirty-two recalled, ninety-six. Compressed: also ninety-six. Hierarchical: seventy-two. The experiment compared mechanisms while quietly handing each one a different allowance. If local-plus-recall wins, is its recall clever, or is it just richer? So the budgets were matched: every variant now gets ninety-six slots at the final query, and every sparsity knob became an explicit command-line flag so the budget is a choice, not an accident.

## 05:58.22-06:58.75
A: So the race was re-run properly. One pass over a fifty-million-character mix of prose, news, code, and LaTeX. Six layers, four point nine million parameters, forty-nine million tokens per run, two independent seeds. And now the four curves finally tell a story. Both seeds agree on exactly the same order. Local-plus-recall finishes best, around one point three one nats. The oracle sits just behind it. The two compressed variants trail by twenty to thirty thousandths of a nat, and that gap is more than twice the largest seed-to-seed wobble. Read it carefully, though. Local-plus-recall versus the oracle is still too close to call; their gap hides inside the seed noise. But token-level selection versus compressed memory is a real, repeatable difference. At equal budget, keeping real tokens beats keeping blurred summaries. That is a conclusion the rigged race could never have earned.

## 06:58.75-07:51.99
A: Boundaries, stated plainly. Five million parameters is a toy. Byte-level loss is not benchmark accuracy. The top-k variant is still an oracle, an upper bound, not a shipping mechanism. And all four run as dense simulations, so their throughput numbers measure masking overhead, not real sparse kernels. But the lesson generalizes ruthlessly. An ablation is a measuring instrument. Before you read the dial, you calibrate it: prove causality with gradients, make silent failure modes unrepresentable, and above all, match the budgets, because a comparison that varies two things at once answers neither question. The most dangerous experiment is not the one that crashes. It is the one that runs perfectly and measures the wrong thing.
