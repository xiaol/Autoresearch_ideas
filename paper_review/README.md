# PAT-Style Automated Paper Review Tool

A multi-level automated paper review tool inspired by Google's **Paper Assistant Tool (PAT)**
(arXiv:2606.28277). It applies **inference scaling** — running progressively deeper passes —
to catch claim-vs-evidence mismatches, benchmark confounds, and methodological issues
in LaTeX-formatted papers.

Built for the `Autoresearch_ideas` research pipeline. The same class of issues this tool
detects was discovered manually in `docs/07_higher_order_benchmark_audit.md`.

## Quick Start

```bash
# Run a surface-level scan
python -m paper_review.cli papers/associative_state_universal_transformers/main.tex --level 1

# Run claim extraction and experiment cross-referencing
python -m paper_review.cli papers/recurrent_ffn/main.tex --level 2

# Full deep audit (claims + benchmark methodology + math)
python -m paper_review.cli papers/recurrent_ffn/main.tex --level 3

# Full verification
python -m paper_review.cli papers/recurrent_ffn/main.tex --level 4

# List all discoverable papers
python -m paper_review.cli --list-papers

# Generate a Markdown report instead of JSON
python -m paper_review.cli papers/associative_state_universal_transformers/main.tex --level 2 --format md
```

## Review Levels

| Level | Name | What it checks | PAT Analog |
|---|---|---|---|
| 1 | Surface Scan | Section structure, equation/table/figure counts, citation counts | Level 1 — AI as formatting checker |
| 2 | Claim Extraction | Numerical claims, comparisons, benchmark results, cross-referenced against experiment outputs | Level 2 — AI-assisted fact-checking |
| 3 | Deep Audit | Benchmark methodology, task identifiers, baseline fairness, confounds | Level 3 — AI-augmented review |
| 4 | Full Verification | Math content analysis, equation/theorem/proof counts, reference completeness | Level 4 — AI-led verification |

## Tool Structure

```
paper_review/
  __init__.py          Package init
  cli.py               CLI entry point (argparse)
  config.py            Review level definitions and configuration
  core.py              Review orchestration engine (multi-pass)
  claim_extractor.py   LaTeX claim extraction (numerical, comparison, benchmark)
  experiment_validator.py  Cross-reference claims against results/ JSON
  benchmark_auditor.py     Benchmark methodology and confound checks
  math_checker.py      Equation/theorem/proof identification
  report.py            Structured JSON + Markdown report generation
  README.md            This file
```

## Input

- **Primary target**: LaTeX `.tex` files from paper packages in `papers/`
- **Experiment data**: JSON result files in `results/` for claim cross-referencing

## Output

- JSON report (default) or Markdown report
- Printed summary to stdout

## Output Format

Each report includes:

- **Paper info**: source path, review level, timestamp
- **Section structure**: sections found, equation/table/figure/citation counts
- **Claims**: extracted numerical, comparison, benchmark, and parameter claims
- **Validation**: cross-referenced claim-vs-experiment mismatches
- **Audit**: benchmark methodology findings (task IDs, metric definitions, confounds)
- **Math report**: equation, theorem, proof counts; unresolved references

## Source Paper

This tool is inspired by:

> Rajesh Jayaram, Drew Tyler, David Woodruff, Corinna Cortes, Yossi Matias,
> Vahab Mirrokni, Vincent Cohen-Addad. "Towards Automating Scientific Review
> with Google's Paper Assistant Tool." arXiv:2606.28277, 2026.
> https://arxiv.org/abs/2606.28277
