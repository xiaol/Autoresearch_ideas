# Qwen/RWKV Trace Comparison Video

This folder contains the narrated screen recording for the interactive
Transformer-vs-RWKV visualization.

## Artifact

- `qwen_rwkv_trace_comparison_plus20db.mp4`: final narrated MP4 with subtitles and a plus-20 dB narration boost.
- `qwen_rwkv_trace_comparison.srt`: subtitle track used in the final mux.
- `narration_beats.json`: timestamped narration map aligned to the right-bottom tutorial card sequence.
- `narration_script.md`: short description of the narration structure.
- `render_report.md`: source media, TTS, mux, and loudness report.
- `tts_timing_report.json`: per-beat TTS duration and speed report.
- `contact_sheet_20s.jpg`: full-screen video contact sheet every 20 seconds.
- `textbook_card_sheet.jpg`: contact sheet focused on the reading/tutorial cards.
- `rwkv_card_sheet.jpg`: contact sheet focused on the RWKV tutorial-card section.

## Source

- Source recording: `/home/xiaol/2026-06-03 23-02-00.mp4`
- Working source app: `/home/xiaol/X/transformer-explainer-qwen`
- Narration provider: MiniMax `speech-2.8-hd`
- Voice: `English_CaptivatingStoryteller`

## Coverage

The first half follows the Qwen 0.8B Transformer trace: embeddings,
blocks, Q/K/V, masked self-attention, MLP, logits, probabilities,
sampling, residuals, normalization, and dropout.

The second half follows the RWKV-7 0.1B recurrent trace: Time Mix,
recurrent state update, matrix and manifold state calculation views,
Channel Mix, logits/probabilities, residuals, and normalization.
