---
name: ml-training-diagnostics
description: Diagnose unstable or confusing machine learning training behavior such as oscillating loss, overfitting, stalled validation, divergence, or suspiciously good training results. Use when a learner or engineer has logs, curves, symptoms, or training observations and needs ranked hypotheses plus low-cost next diagnostics.
---

# ML Training Diagnostics

Use this skill when training behavior is confusing and the next step is not obvious.

## What This Skill Does

This skill turns symptoms into a ranked debugging process.

It is useful for:

- oscillating or diverging loss
- validation stagnation
- fast overfitting
- train / validation mismatch
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

## Quality Bar

- Do not jump straight to architecture changes.
- Check learning rate, scale, split quality, and label quality early.
- Prefer cheap diagnostics before expensive reruns.
- Preserve experiment attribution whenever possible.

## Good Prompt Shapes

- My training loss oscillates wildly. Diagnose it.
- Validation is flat but training accuracy rises. What are the top causes?
- Rank the most likely reasons this model is overfitting.

## Reference

Read [references/symptom-map.md](references/symptom-map.md) for a quick symptom-to-hypothesis map.
