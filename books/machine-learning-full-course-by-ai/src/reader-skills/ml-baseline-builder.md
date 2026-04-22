# Reader Skill: ML Baseline Builder

This page mirrors the real local skill used in the repository.

Use it after [Chapter 4](../chapter-04/README.md) whenever a modeling effort is drifting toward unnecessary complexity before simple baselines are understood.

Real skill source:

`.agents/skills/ml-baseline-builder/SKILL.md`

Invoke in Codex:

`$ml-baseline-builder`

## What This Skill Does

This skill produces:

1. The simplest naive baseline
2. The first interpretable baseline
3. A shared comparison metric
4. Slice checks
5. One reason to escalate complexity
6. One reason not to escalate complexity yet

## Workflow

1. Define the task and target clearly.
2. Choose the naive baseline.
3. Choose the first interpretable baseline.
4. Compare them on the same split and metric.
5. Inspect failure patterns by slice.
6. State whether a richer model is justified.

## Output Format

- `Task`
- `Naive Baseline`
- `Interpretable Baseline`
- `Comparison Metric`
- `Critical Slices`
- `Why Complexity May Be Justified`
- `Why Complexity May Not Be Justified`
- `Recommendation`

## Try It Now

```text
Use $ml-baseline-builder to design a first comparison plan for a delivery-time prediction task. Include the naive baseline, the first interpretable baseline, the shared comparison metric, critical slices, and one reason a more complex model may or may not be justified.
```
