# Generic Triple-Latent Paper Package

This folder is the standalone package for the paper:

- `generic_triple_latent.tex`: main paper source
- `refs.bib`: bibliography
- `neurips_2025.sty`: style file
- `figures/generic_triple_latent_overview.pdf`: architecture figure used in the paper
- `figures/generic_triple_latent_overview.png`: preview image of the architecture figure
- `figures/render_generic_triple_latent_figure.py`: figure generator
- `supporting_results/`: copied JSON/CSV artifacts for the paper tables
- `generic_triple_latent.pdf`: latest rebuilt paper
- `generic_triple_latent_final.pdf`: finalized paper artifact

Build from this folder with:

```bash
latexmk -pdf generic_triple_latent.tex
```

Notes:

- This package is intentionally self-contained for handoff and archival.
- The broader experiment code remains in the `auto_research_llm_ideas` project tree.
- The new paper tables are backed by these repo scripts:
  - `auto_research_llm_ideas/experiments/run_generic_triple_ruler.py`
  - `auto_research_llm_ideas/experiments/estimate_generic_triple_compute.py`
- Follow-up experiments after the paper snapshot are backed by:
  - `auto_research_llm_ideas/experiments/run_stabilized_gated_recall.py`
  - `auto_research_llm_ideas/experiments/run_long_context_transfer.py`
