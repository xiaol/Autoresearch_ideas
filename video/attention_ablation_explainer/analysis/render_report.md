# Render Report — AttentionAblationExplainer (final)

## Final deliverable
- **MP4**: `outputs/AttentionAblationExplainer_4k_minimax_narrated.mp4` (copy at project root)
- 3840x2160 @ 30 fps, h264 + AAC 160k + embedded `mov_text` subtitles (eng), 471.98 s (7:52), 35 MB
- Sidecar SRT: `outputs/AttentionAblationExplainer_4k_minimax_narrated.srt` (12 cues incl.
  end-sentinel guarding the mov_text last-cue drop)
- Silent 4K visual pass: `renders/videos/attention_ablation_explainer/2160p30/AttentionAblationExplainer.mp4`

## Narration / TTS
- MiniMax `speech-2.8-hd`, voice `English_captivating_female1`, single narrator, speed 1.0
- **Narration-first**: script written → per-beat natural durations measured
  (`scripts/measure_tts.py` → `narration/natural_durations.json`) → windows = natural + 3 s hold
  (+2 s closing) via `scripts/apply_windows.py` → scene timeline coded to those windows
- **max_speed_factor = 1.0** (atempo forbidden in `fit_segment`; would hard-fail instead)
- Audio 471.98 s vs video 472.0 s; loudnorm I=-20 per turn, per segment, and final mix
- Intentional holds: 3 s x 11 beats ≈ 7 % of runtime; no unintended gap > 3 s (audio continuous
  by construction: each window = its own natural speech + hold)

## Timeline (absolute ends)
41.05 / 72.63 / 113.68 / 154.19 / 188.51 / 232.26 / 280.51 / 312.06 / 358.22 / 418.75 / 471.99
(11 beats: cold open, testbed, DSA, LSA, CSA+HCA, causality audit, uniform-softmax leak,
instrument errors, budget confound, fixed rerun results, boundaries+synthesis)

## Gates
| gate | status |
|---|---|
| Layout containment (`register_inside`, MANIM_LAYOUT_STRICT=1) | PASS — 15 registered pairs, 0 overflow (`analysis/layout_report.json`) |
| Duration (rendered == narration total) | PASS — 471.99 s both; delta < 0.05 % |
| Voice 1.0x (no atempo) | PASS — max_speed_factor 1.0 |
| Stage-clear at transitions | PASS — `clear_scene()` per beat; post-boundary frames show no stale mobjects (`analysis/final_transition_contact_sheet.png`) |
| Caption strategy | embedded mov_text + sidecar SRT; scene renders no hard-coded caption bars, so no duplication |
| Coverage (reveals map to narration) | PASS — `beat_sync()` word-proportional sync points per reveal (see scene comments) |
| Audience expectation | PASS — mask grids, budget bars, loss curves, -1e9 row, results table drive the frames (contact sheets) |
| Knowledge growth | `analysis/knowledge_growth_ledger.md`; flip beats: uniform-softmax leak, budget morph |
| Term gate | `narration/term_ledger.md` — every load-bearing term defined before use |
| Source gate | all claims from local code, check-suite output, and run metrics (no external claims) |

## QA artifacts
- Mid-beat contact sheet: `analysis/preview_contact_sheet.png` (14 frames)
- Fix-verification sheet: `analysis/fix_check_sheet.png` (3 layout defects found in preview QA
  → fixed: held-fixed/vocab panel overlap, LSA note clipping risk, results verdict placement)
- Transition sheet: `analysis/final_transition_contact_sheet.png` (12 post-boundary frames)
- ffprobe: streams h264/aac/mov_text confirmed (see above)

## Data sources animated
- `attention_pretrain_ablation/train_attention_variants.py` (mechanisms, before/after fixes)
- `check_attention_correctness.py` 68-check suite (causality table beat)
- Baseline run `four_attention_tiny_1k_*` (cold-open 0.014-nat spread)
- `analysis/comparison.json` from `budget96_6k_seed{1337,2024}` (results curves + table,
  drawn programmatically in-scene)

## Deviations / limitations
- Produced by the main thread rather than per-part subagent workers: subagents were unstable
  during this session (stalls + gateway 503s in the review workflow); all worker gates were
  still applied by the producer. Single narrator voice (no B voice) by design.
- Beat-internal sync points are word-proportional estimates (±1–2 s), not forced-aligned.
- The 18x18 mask grid is an explanatory abstraction of the real 256x256 mask (labeled on screen).
