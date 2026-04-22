---
name: ml-tabular-model-review
description: Compare tabular machine learning model families honestly across naive, linear, tree, and ensemble baselines while checking slice behavior, leakage risk, calibration, and practical cost. Use when a team is evaluating structured data models and needs a grounded recommendation instead of novelty-driven model choice.
---

# ML Tabular Model Review

Use this skill when a tabular problem needs an honest model-family comparison rather than a prestige contest.

## What This Skill Does

This skill reviews:

- naive baseline performance
- linear baseline performance
- tree baseline performance
- ensemble performance
- slice comparison
- leakage notes
- recommendation with justification

## Workflow

1. Freeze the split strategy.
2. Review the naive baseline.
3. Review the interpretable linear baseline if applicable.
4. Review a tree baseline.
5. Review an ensemble result.
6. Compare performance by slice, not just globally.
7. Decide whether added complexity changes useful decisions.

## Output Format

Use this structure:

- `Task`
- `Split Strategy`
- `Naive Baseline`
- `Linear Baseline`
- `Tree Baseline`
- `Ensemble Result`
- `Slice Comparison`
- `Leakage Notes`
- `Calibration / Threshold Notes`
- `Recommendation`

## Quality Bar

- Do not compare models on different splits.
- Do not treat one global metric as the whole story.
- Treat leakage risk as part of the review, not a side note.
- Include runtime or complexity cost when the gain is small.

## Good Prompt Shapes

- Review these tabular model results for fraud detection.
- Compare linear, tree, and boosted models for this risk task.
- What should we conclude from this tabular benchmark table?

## Reference

Read [references/review-table-template.md](references/review-table-template.md) for the comparison table template.
