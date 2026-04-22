# Reader Skill: ML Evaluation Review

This page mirrors the real local skill used in the repository.

Use it after [Chapter 6](../chapter-06/README.md) whenever someone presents model metrics and you need to decide whether the result is informative, misleading, or not yet trustworthy.

Real skill source:

`.agents/skills/ml-evaluation-review/SKILL.md`

Invoke in Codex:

`$ml-evaluation-review`

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

## Try It Now

```text
Use $ml-evaluation-review to review this fraud model result: AUC 0.94, recall 0.61 at the current threshold, precision 0.12, and weak performance on new merchants. Tell me what the main metric hides, whether thresholding or calibration matters, which slices are critical, and whether the experiment supports the claim that the new model is better.
```

## Quality Bar

- Do not praise a metric without naming its tradeoff.
- Do not treat accuracy as meaningful by default on imbalanced tasks.
- Do not ignore threshold choice when score-to-action matters.
- Always ask what evidence would weaken confidence in the result.

## Reference Checklist

Ask:

1. What decision is the model supporting?
2. Does the metric reflect that decision?
3. Is the task imbalanced?
4. Do thresholds matter operationally?
5. Does calibration matter?
6. Which slices are high risk?
7. Can the reported result be trusted given the experiment design?
