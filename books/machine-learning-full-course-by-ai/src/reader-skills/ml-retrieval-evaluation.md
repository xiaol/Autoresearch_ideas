# Reader Skill: ML Retrieval Evaluation

This page mirrors the real local skill used in the repository.

Use it after [Chapter 11](../chapter-11/README.md) whenever search quality should be judged by evidence rather than plausible demos.

Real skill source:

`.agents/skills/ml-retrieval-evaluation/SKILL.md`

Invoke in Codex:

`$ml-retrieval-evaluation`

## What This Skill Does

This skill reviews:

- retrieval metric summary
- representative good examples
- representative bad examples
- failure pattern categories
- recommendation for representation, indexing, or ranking changes

## Workflow

1. Define what relevant means for the task.
2. Build or inspect a small evaluation query set.
3. Measure retrieval performance at practical cutoffs.
4. Inspect top failures manually.
5. Decide whether the main problem is representation, indexing, or ranking.

## Output Format

- `Task`
- `Relevance Definition`
- `Query Set Quality`
- `Metric Summary`
- `Representative Good Results`
- `Representative Bad Results`
- `Failure Pattern Categories`
- `Representation vs Indexing vs Ranking Diagnosis`
- `Recommended Next Change`

## Try It Now

```text
Use $ml-retrieval-evaluation to review a company knowledge search system. Define what relevance means, summarize practical retrieval metrics, show representative good and bad results, classify failure patterns, and decide whether the main issue is representation, indexing, or ranking.
```
