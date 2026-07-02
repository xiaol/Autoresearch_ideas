# Attention Ablation Explainer — Manim video

Explainer for the `~/X/attention_pretrain_ablation/` experiment: four sparse-attention variants
(DSA oracle top-k / LSA local+recall / CSA / HCA compressed), the audit that found a latent
uniform-softmax future-leak, harness bugs, and an unmatched-budget confound — then the
budget-matched 49M-token rerun (2 seeds) where token-level selection repeatably beats
compressed memory.

## Layout
- `scenes/attention_ablation_explainer.py` — single-scene global timeline, 11 beats,
  layout-strict (`register_inside` + `MANIM_LAYOUT_STRICT=1`); beat end-times loaded from
  `narration/beat_windows.json`; results drawn from `analysis/comparison.json`
- `narration/narration_tts.md` — voice-ready script; windows = measured natural MiniMax TTS
  duration + 3s hold (apply via `scripts/apply_windows.py`)
- `narration/beat_map.md`, `narration/term_ledger.md`, `analysis/knowledge_growth_ledger.md`
- `scripts/measure_tts.py` — synthesize + measure natural per-beat durations (cached)
- `scripts/apply_windows.py` — write windows from naturals (voice never sped up; atempo forbidden)
- `scripts/generate_minimax_narration.py` — TTS + loudnorm + SRT + mux (fails hard if any
  segment audio exceeds its window)

## Build
```bash
CONDA=/home/xiaol/X/ai_hunt_replicate/.conda-manim/bin
$CONDA/python scripts/measure_tts.py && $CONDA/python scripts/apply_windows.py
MANIM_PREVIEW=1 MANIM_LAYOUT_STRICT=1 $CONDA/manim -ql --disable_caching \
    scenes/attention_ablation_explainer.py AttentionAblationExplainer   # preview + layout gate
MANIM_LAYOUT_STRICT=1 $CONDA/manim -qh --disable_caching \
    scenes/attention_ablation_explainer.py AttentionAblationExplainer   # 4K
$CONDA/python scripts/generate_minimax_narration.py                     # TTS + mux + SRT
```

Final: `outputs/AttentionAblationExplainer_4k_minimax_narrated.mp4` (+ sidecar `.srt`,
report `outputs/minimax_narration_report.json`, QA in `analysis/`).
