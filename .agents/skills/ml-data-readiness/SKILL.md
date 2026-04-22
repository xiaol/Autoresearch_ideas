---
name: ml-data-readiness
description: Assess whether a machine learning dataset and feature pipeline are trustworthy enough for serious modeling by reviewing schemas, timestamps, freshness, leakage risks, joins, lineage, and train-serve consistency. Use when data quality is uncertain, upstream changes may have occurred, or a team wants a data-readiness review before training.
---

# ML Data Readiness

Use this skill before trusting a dataset just because it loaded successfully.

## What This Skill Does

This skill reviews whether the data and feature path are ready for serious modeling.

It checks:

- schema meaning
- timestamp semantics
- joins and duplication risks
- freshness
- leakage
- train-serve mismatches
- lineage and ownership

## Workflow

1. List the sources and critical fields.
2. Clarify timestamp meaning and prediction-time availability.
3. Check for duplication, joins, and missingness anomalies.
4. Review freshness and train-serve consistency.
5. State what is trustworthy and what remains uncertain.

## Output Format

Use this structure:

- `Sources`
- `Critical Fields`
- `Timestamp Semantics`
- `Freshness Risks`
- `Join / Duplication Risks`
- `Leakage Risks`
- `Train-Serve Risks`
- `Lineage / Ownership`
- `Readiness Verdict`
- `Required Fixes`

## Quality Bar

- Treat schema as meaning, not only type.
- Be explicit about prediction-time availability.
- Assume silent joins and feature drift are possible until checked.
- Prefer a narrow honest verdict over a broad unearned one.

## Good Prompt Shapes

- Review whether this fraud dataset is ready for modeling.
- What data risks do you see in this feature pipeline description?
- Check this dataset for leakage and train-serve mismatch risks.

## Reference

Read [references/readiness-checklist.md](references/readiness-checklist.md) for the detailed data review checklist.
