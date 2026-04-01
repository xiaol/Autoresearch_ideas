# Localized Recurrent-FFN Paper

This folder contains the standalone paper draft for the localized recurrent FFN
project.

Main files:

- `main.tex`: standalone draft
- `main.pdf`: compiled PDF
- `refs.bib`: bibliography for this draft
- `neurips_2025.sty`: local style file copy for independent builds

Build command:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

Primary result sources used in the current draft:

- `/Users/xiaol/x/PaperX/lld_paper/results/recurrent_ffn_pilot_v1/summary.json`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_lm_pilot_v1/summary.json`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_method_screen_swiglu_v1/summary.json`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_swiglu_followup_v1/summary.json`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_accuracy_push_v1/summary.json`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_accuracy_push_followup_v1/summary.json`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_ultra_search_v2/summary.json`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_transformer_controls_v2/summary.json`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_ultra_vs_transformer_400_v1/summary.json`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_ultra_vs_transformer_400_seed11_v1/summary.json`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/recurrent_ffn_ultra_throughput_v1/summary.json`

Current paper framing:

- plain Recurrent-FFN is still the synthetic-memory win
- several recurrent variants beat the full-width SwiGLU baseline at 120 steps
- `stable` and `readout` are the strongest fair small-model recurrent variants
- the strongest final result comes from the accuracy-first `recurrent_ffn_ultra`
- `ultra-384` beats a matched-capacity Transformer in a two-seed 400-step pilot
- the recurrent winner is still much slower than the Transformer in this Python
  reference implementation
