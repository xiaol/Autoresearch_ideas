# Softmax Attention to Gated DeltaNet

This package contains an English, first-principles reconstruction of the
derivation in Scientific Spaces archive 11823. It explains how a fixed-query
softmax recurrence exposes an exact Delta-shaped correction, where the local
approximations enter, and why the resulting query-independent matrix update has
the retain--erase--write structure of Gated DeltaNet.

## Files

- `softmax_attention_to_gated_deltanet_en.tex`: editable source
- `softmax_attention_to_gated_deltanet_en.pdf`: compiled explainer
- `refs.bib`: source references

## Build

The source uses XeLaTeX for the Noto typefaces available on the target system.
It otherwise stays within the repository's minimal TinyTeX package set.

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
  softmax_attention_to_gated_deltanet_en.tex
```

## Claim boundary

The source article is analytic and contains no experiments or benchmark table.
The document distinguishes exact identities, Taylor approximations, modeling
assumptions, and the final structural comparison. In particular, it does not
claim that trained Gated DeltaNet is exactly equal to softmax attention.

Primary source: <https://kexue.fm/archives/11823>
