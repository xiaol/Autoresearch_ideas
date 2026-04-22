# Reader Skill: ML Tabular Model Review

This page mirrors the real local skill used in the repository.

Use it after [Chapter 5](../chapter-05/README.md) whenever a structured-data project needs an honest comparison between linear, tree, and ensemble families.

Real skill source:

`.agents/skills/ml-tabular-model-review/SKILL.md`

Invoke in Codex:

`$ml-tabular-model-review`

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

## Try It Now

```text
Use $ml-tabular-model-review to compare naive, linear, tree, and boosted-tree baselines for a fraud detection task. Include split strategy, slice comparison, leakage notes, calibration concerns, and a justified recommendation.
```
