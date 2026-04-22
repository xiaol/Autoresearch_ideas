# Reader Skill: ML Training Diagnostics

This page mirrors the real local skill used in the repository.

Use it after [Chapter 7](../chapter-07/README.md) whenever training behavior is confusing and the next experiment is not obvious.

Real skill source:

`.agents/skills/ml-training-diagnostics/SKILL.md`

Invoke in Codex:

`$ml-training-diagnostics`

## What This Skill Does

This skill turns symptoms into a ranked debugging process.

It is useful for:

- oscillating or diverging loss
- validation stagnation
- fast overfitting
- train and validation mismatch
- unstable gradients
- suspiciously good or suspiciously bad runs

## Workflow

1. Describe the symptom precisely.
2. Separate optimization symptoms from data symptoms.
3. Rank the most likely causes.
4. Propose one low-cost diagnostic for each cause.
5. Recommend the next experiment that preserves attribution.

## Output Format

Use this structure:

- `Observed Symptom`
- `Most Likely Causes`
- `Ranked Hypotheses`
- `Low-Cost Diagnostics`
- `Next Experiment`
- `What Would Falsify This Diagnosis`

## Try It Now

```text
Use $ml-training-diagnostics to diagnose this training run: training accuracy rises quickly, validation accuracy stays flat, and the loss oscillates after the fifth epoch. Rank the likely causes, propose low-cost diagnostics, and recommend the next experiment that preserves attribution.
```

## Quality Bar

- Do not jump straight to architecture changes.
- Check learning rate, scale, split quality, and label quality early.
- Prefer cheap diagnostics before expensive reruns.
- Preserve experiment attribution whenever possible.

## Reference Symptom Map

Oscillating or diverging loss often points to:

- learning rate too high
- unstable gradients
- bad scaling

Training improves while validation collapses often points to:

- overfitting
- split mismatch
- leakage or label problems

When nothing learns, check:

- broken data path
- wrong loss or task setup
- learning rate too low
- bad initialization
