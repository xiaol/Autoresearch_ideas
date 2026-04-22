---
name: ml-retrieval-evaluation
description: Evaluate whether an embedding and retrieval system is useful for a real task by reviewing query quality, relevance definitions, metrics at practical cutoffs, representative failures, and whether problems come from representation, indexing, or ranking. Use when a team has a search or retrieval system and needs inspectable evidence of quality.
---

# ML Retrieval Evaluation

Use this skill when retrieval quality should be judged by evidence rather than plausible-looking demos.

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

Use this structure:

- `Task`
- `Relevance Definition`
- `Query Set Quality`
- `Metric Summary`
- `Representative Good Results`
- `Representative Bad Results`
- `Failure Pattern Categories`
- `Representation vs Indexing vs Ranking Diagnosis`
- `Recommended Next Change`

## Quality Bar

- Do not let pretty examples stand in for evaluation.
- Use practical cutoffs, not only abstract ranking metrics.
- Treat retrieval as a system, not only as an embedding.
- Manual failure inspection is required.

## Good Prompt Shapes

- Review whether this company search system is actually working.
- Diagnose retrieval failures for these support queries.
- Is the main problem our embeddings, indexing, or reranking?

## Reference

Read [references/retrieval-review-sheet.md](references/retrieval-review-sheet.md) for the review-sheet template.
