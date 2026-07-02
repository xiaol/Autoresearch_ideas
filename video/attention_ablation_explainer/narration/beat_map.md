# Beat Map — Attention Pretrain Ablation Explainer

Windows = measured natural MiniMax duration (speech-2.8-hd, English_captivating_female1, 1.0x)
plus an intentional 2.5–3s breathing hold per beat. Final absolute windows are written by
scripts/apply_windows.py into narration_tts.md + narration/beat_windows.json after the results
beat (10) is written and measured. Voice is NEVER sped up; scene stretches to the voice.

| # | beat | natural TTS | hold | on-screen carrier | visual action | viewer learns | source |
|---|------|------------|------|-------------------|---------------|---------------|--------|
| 1 | cold open: rigged race | 38.05s | ~3s | 4 mask thumbnails + old val losses | thumbnails in, brace "gap 0.014 nats", strike through reading 1 | near-identical losses can mean broken instrument, not equal mechanisms | four_attention_tiny_1k summary.json |
| 2 | testbed, one knob | 28.58s | ~2.5s | byte-chip stream + 6-layer stack | vocab-257 panel; amber knob box around attention slots; "held fixed" panel | what a byte-level LM ablation holds fixed | train_attention_variants.py ModelConfig |
| 3 | DSA oracle top-k | 38.05s | ~3s | 18x18 causal grid | query row scores all past -> top-k cells; formula; ORACLE panel | top-k here = perfect-selection upper bound, not a deployable indexer | DSAAttention (lines ~148-168) |
| 4 | LSA local+recall | 37.51s | ~3s | same grid archetype | local band; block rectangles; chosen block -> fine tokens; layer-reuse chips | two-tier selection + cross-layer mask reuse | LSAAttention.build_mask, TinyLM.forward |
| 5 | CSA/HCA compressed | 31.32s | ~2.5s | token->gate->summary pipeline | gated pooling; ONE softmax panel [local ‖ compressed]; CSA-vs-HCA panel | compressed KV competes in the same softmax as crisp tokens | CompressedAttention.compress/forward |
| 6 | causality audit | 40.75s | ~3s | dy_i/dx_j = 0 law | law payoff; 4x5 PASS table; 68-check tally | causality is a gradient-testable property, exactly zero | check_attention_correctness.py |
| 7 | uniform-softmax leak | 45.25s | ~3s | all -1e9 score row | row -> softmax(equal)=1/n -> uniform bars, future bars red; danger panel | all-masked row = uniform attn = silent future leak; fix = unrepresentable | masked_topk_mask + ModelConfig.__post_init__ |
| 8 | instrument errors | 28.55s | ~2.5s | dataset strip + metrics log | last window red/unreachable; randint code; interleaved jsonl lines; fixes panel | off-by-one + append-mode contamination corrupt records silently | get_batch, train_variant |
| 9 | budget confound | 43.16s | ~3s | 4 budget bars 64/96/96/72 | bars in; "clever or richer?"; dsa+hca morph to 96; flags line | unmatched budgets confound mechanism comparisons; match then compare | ModelConfig defaults before/after |
| 10 | fixed rerun results | TBD | ~3s | loss-vs-tokens axes + table | 4 mean curves (2 seeds); table final val ± spread; gap-vs-spread verdict | whether mechanism gaps exceed seed noise at 49M tokens | budget96_6k_seed{1337,2024}/comparison.json |
| 11 | boundary + synthesis | 48.24s | ~3s hold at end | boundaries panel + 3 checklist chips | chips in sequence; closing two-liner | calibration checklist; runs-perfectly-but-wrong is the real danger | this audit |

Total ≈ 38+28.6+38.1+37.5+31.3+40.8+45.3+28.6+43.2+[seg10]+48.2 ≈ 380s + seg10 + holds ≈ ~7.5 min.
