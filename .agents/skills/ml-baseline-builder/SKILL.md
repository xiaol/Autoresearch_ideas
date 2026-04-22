---
name: ml-baseline-builder
description: Build an honest first model comparison for a machine learning task by defining a naive baseline, an interpretable baseline, shared metrics, slice checks, and the evidence needed before escalating complexity. Use when a team is skipping baselines, jumping to complex models too early, or needs a one-page recommendation for what model family to try next.
---

# ML Baseline Builder

Use this skill when a modeling effort needs disciplined starting points rather than model-fashion momentum.

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

Use this structure:

- `Task`
- `Naive Baseline`
- `Interpretable Baseline`
- `Comparison Metric`
- `Critical Slices`
- `Why Complexity May Be Justified`
- `Why Complexity May Not Be Justified`
- `Recommendation`

## Quality Bar

- Do not recommend complexity before a naive and interpretable baseline exist.
- Keep the comparison on the same split and metric.
- Name at least one slice where the baselines may fail differently.
- Treat simple models as instruments for understanding, not as embarrassment.

## Good Prompt Shapes

- Build a baseline plan for delivery-time prediction.
- What baselines should we try before a deep model on this support-ticket task?
- Turn this project brief into a one-page baseline comparison memo.

## Reference

Read [references/comparison-memo-template.md](references/comparison-memo-template.md) for the comparison memo template and baseline checklist.
