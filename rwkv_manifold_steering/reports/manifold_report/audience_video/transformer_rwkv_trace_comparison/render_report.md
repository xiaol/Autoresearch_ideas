# Narration Render Report

Source video:

- `/home/xiaol/2026-06-03 23-02-00.mp4`
- Duration: `484.933333s`
- Video: H.264, `1920x1080`, `60 fps`
- Original audio: present but effectively silent (`mean_volume=-91.0 dB`)

Generated outputs:

- Final narrated MP4: `/home/xiaol/X/transformer-explainer-qwen/video_narration/qwen_rwkv_narrated.mp4`
- Louder narrated MP4: `/home/xiaol/X/transformer-explainer-qwen/video_narration/qwen_rwkv_narrated_louder.mp4`
- Plus-20 dB narrated MP4: `/home/xiaol/X/transformer-explainer-qwen/video_narration/qwen_rwkv_narrated_plus20db.mp4`
- Narration WAV: `/home/xiaol/X/transformer-explainer-qwen/video_narration/narration.wav`
- Subtitles: `/home/xiaol/X/transformer-explainer-qwen/video_narration/narration.srt`
- Beat map: `/home/xiaol/X/transformer-explainer-qwen/video_narration/narration_beats.json`
- TTS timing report: `/home/xiaol/X/transformer-explainer-qwen/video_narration/tts_timing_report.json`

TTS:

- Provider: MiniMax
- Model: `speech-2.8-hd`
- Voice: `English_CaptivatingStoryteller`
- Voice query confirmation: `minimax_voice_query.json`
- Audio generation: per-beat MP3, normalized with `loudnorm=I=-20:TP=-1.5:LRA=11`
- Max speed factor: `1.0006042296072521`

Final mux verification:

- Video stream: H.264, `1920x1080`, `60 fps`
- Audio stream: AAC, `48000 Hz`, mono
- Subtitle stream: `mov_text`
- Final duration: `484.933333s`
- Final size: about `59 MB`

Louder remux:

- Audio filter: `volume=6dB,alimiter=limit=0.95`
- Audio stream: AAC, `48000 Hz`, mono, `224k`
- Peak after boost: `max_volume=-0.6 dB`
- Duration: `484.949s`

Plus-20 dB remux:

- Audio filter: `volume=20dB,alimiter=limit=0.95`
- Audio stream: AAC, `48000 Hz`, mono, `224k`
- Mean after boost: `mean_volume=-19.9 dB`
- Peak after boost: `max_volume=-0.0 dB`
- Duration: `484.949s`

Leveled final remux:

- Reason: the raw plus-20 dB remux had a visible loudness ramp. Spoken-window means moved from about `-27.4 dB` at the beginning to about `-11.8 dB` at the end.
- Audio filter strategy: per-tutorial-beat gain correction, `alimiter=limit=0.90`, `loudnorm=I=-18:TP=-1.5:LRA=7`, then `aresample=48000`.
- Audio stream: AAC, `48000 Hz`, mono, `224k`.
- Spoken-window mean after leveling: `-19.6 dB` to `-18.6 dB`, about a `1.0 dB` span across the 19 narration beats.
- Output path in this repo: `qwen_rwkv_trace_comparison_plus20db.mp4`. The filename is kept for compatibility with the first pushed artifact, but the audio is the leveled remux.

Narration coverage:

- Qwen 0.8B sequence follows the Transformer textbook pages: embeddings, blocks, Q/K/V, attention, MLP, logits/probabilities, sampling, residuals, normalization, dropout.
- RWKV-7 0.1B sequence follows the RWKV textbook pages: recurrent model framing, Time Mix, recurrent state, matrix/manifold state calculation, Channel Mix, probabilities, residuals, normalization.
