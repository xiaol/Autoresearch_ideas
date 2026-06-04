# Qwen/RWKV Transformer-RNN Explainer

This project adapts Polo Club's Transformer Explainer into an interactive visualization for explaining Transformer-style and RNN-style LLM internals.

It compares:

- Qwen 0.8B as a small Transformer trace.
- RWKV-7 0.1B as a recurrent state-based, RNN-style LLM trace.

The goal is to make the difference between masked self-attention and recurrent state transition visible through the same tutorial surface.

## Contents

- `src/`: Svelte app, visualization components, model-aware tutorial cards, Qwen attention views, and RWKV state/matrix/manifold views.
- `static/qwen-traces/`: saved Qwen example traces.
- `static/rwkv-traces/`: saved RWKV-7 example traces.
- `scripts/`: trace-generation scripts for Qwen and RWKV.
- `video_narration/`: narrated comparison video, subtitle file, contact sheets, narration beats, and render report.

The old browser GPT-2 ONNX chunk files from the upstream project are not included under `static/model-v2/`. This package is focused on the static Qwen/RWKV trace visualization.

## Run Locally

```bash
npm install
npm run dev
```

Then open the local Vite URL, usually:

```text
http://localhost:5173
```

The included example prompts use the saved trace files. Custom prompts call the local SvelteKit API routes and require the Python model environments used by `scripts/qwen_trace.py` and `scripts/rwkv_trace.py`.

## Narrated Video

The current video artifact is:

```text
video_narration/qwen_rwkv_trace_comparison_plus20db.mp4
```

The audio was leveled per tutorial beat and the closing RWKV section was tapered down so the end does not overpower the beginning.

Suggested YouTube direction:

```text
Explaining Transformer and RNN-Style LLMs: Qwen vs RWKV-7 Visualized
```

## Upstream

This is based on Polo Club's Transformer Explainer:

```text
https://github.com/poloclub/transformer-explainer
```

The upstream project and this adapted visualization are MIT licensed.
