---
name: ml-evaluation-review
description: Review whether a machine learning result is actually trustworthy by checking metric-task fit, class imbalance, slice failures, thresholding, calibration, and experiment attribution. Use when someone presents model metrics, evaluation charts, or experiment results and needs an honest engineering review.
---

# ML Evaluation Review

Use this skill to pressure-test whether a model result means what the team thinks it means.

## What This Skill Does

This skill reviews:

- whether the chosen metric fits the task
- what the metric may be hiding
- whether class imbalance changes interpretation
- whether thresholds or calibration matter
- whether slice failures undermine the global result
- whether the experiment design supports the claimed conclusion

## Workflow

1. Restate the task and business target.
2. Identify the primary metric and its blind spots.
3. Check class balance or error distribution.
4. Review thresholds, score usage, and calibration needs.
5. Ask which slices matter most.
6. Ask whether the experiment changed one major thing or many.
7. Decide what can actually be concluded.

## Output Format

Use this structure:

- `Task`
- `Main Metric`
- `What This Metric Hides`
- `Threshold / Calibration Notes`
- `Critical Slices`
- `Experiment Design Risk`
- `Trust Level`
- `Recommended Next Step`

## Quality Bar

- Do not praise a metric without naming its tradeoff.
- Do not treat accuracy as meaningful by default on imbalanced tasks.
- Do not ignore threshold choice when score-to-action matters.
- Always ask what evidence would weaken confidence in the result.

## Good Prompt Shapes

- Review these fraud model results and tell me if they are deployment-ready.
- Does this evaluation actually support the claim that the new model is better?
- What is missing from this experiment summary?

## Reference

Read [references/review-checklist.md](references/review-checklist.md) for the detailed evaluation checklist.
