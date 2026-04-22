---
name: ml-ranking-review
description: Review a recommendation or ranking system as a decision loop by examining objective choice, metric stack, feedback loops, exploration sufficiency, offline-online mismatch, and rollout readiness. Use when ranking gains look promising but the team needs an honest system-level review before rollout or redesign.
---

# ML Ranking Review

Use this skill when a recommendation or ranking system should be reviewed as an intervention system, not just a predictor.

## What This Skill Does

This skill reviews:

- the primary objective and its limits
- the metric stack
- feedback loop risks
- whether exploration is sufficient
- whether offline gains are likely to hold online
- rollout, redesign, or testing recommendation

## Workflow

1. State what the system is optimizing.
2. Ask what that objective leaves out.
3. Identify who could be overexposed or underexposed.
4. Review offline and online evidence separately.
5. Decide whether the system is learning healthy or distorted behavior.

## Output Format

Use this structure:

- `System Objective`
- `What the Objective Leaves Out`
- `Metric Stack`
- `Feedback Loop Risks`
- `Exploration Sufficiency`
- `Offline vs Online Evidence`
- `Exposure / Fairness Concerns`
- `Recommendation`

## Quality Bar

- Do not reduce the system to one engagement metric.
- Separate predictive quality from behavioral quality.
- Name at least one feedback loop risk explicitly.
- Include exploration and exposure concerns in the review.

## Good Prompt Shapes

- Review this recommendation system before rollout.
- Do these ranking gains actually matter for the live product?
- What is missing from this ranking system review?

## Reference

Read [references/ranking-memo-template.md](references/ranking-memo-template.md) for the ranking review memo.
