# Associative-State Universal Transformers

This folder contains the self-contained manuscript package for the main
UniMatrix paper.

Main files:

- `main.tex`: primary manuscript
- `main.pdf`: compiled manuscript PDF
- `refs.bib`: bibliography used by the manuscript
- `neurips_2025.sty`: local style file copy for independent builds
- `figures/`: local copies of paper figures used by the manuscript

Additional draft:

- `generic_triple_latent.tex`: related standalone draft kept with the same
  paper package because it shares bibliography and figure assets

Build command:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Primary figure sources mirrored into `figures/`:

- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/assets/arch/unimatrix_paper_overview.pdf`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/assets/arch/generic_triple_latent_overview.pdf`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/pilot_neurips/figures/lm_validation_bpb.png`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/results/pilot_neurips/figures/throughput_scaling.png`

Paper-specific figure renderers now target this folder directly:

- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/tools/render_paper_figure1.py`
- `/Users/xiaol/x/PaperX/auto_research_llm_ideas/assets/arch/render_generic_triple_latent_figure.py`
