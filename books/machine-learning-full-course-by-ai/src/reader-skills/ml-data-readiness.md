# Reader Skill: ML Data Readiness

This page mirrors the real local skill used in the repository.

Use it after [Chapter 13](../chapter-13/README.md) whenever a dataset looks convenient but you do not yet trust its timestamps, joins, freshness, or train-serve path.

Real skill source:

`.agents/skills/ml-data-readiness/SKILL.md`

Invoke in Codex:

`$ml-data-readiness`

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

## Try It Now

```text
Use $ml-data-readiness to review a delivery-time prediction dataset built from orders, courier events, weather snapshots, and customer support logs. Check timestamp semantics, feature freshness, join risks, leakage, train-serve mismatches, and ownership of the critical features.
```

## Quality Bar

- Treat schema as meaning, not only type.
- Be explicit about prediction-time availability.
- Assume silent joins and feature drift are possible until checked.
- Prefer a narrow honest verdict over a broad unearned one.

## Reference Checklist

Ask:

1. What creates a row?
2. What timestamp defines prediction time?
3. Which fields are known at that moment?
4. Which joins could duplicate or distort data?
5. Which features may be stale?
6. Which fields may leak future information?
7. Is the serving path equivalent to the training path?
8. Who owns the critical features?
