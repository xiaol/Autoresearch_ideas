# Reader Skill: ML Ranking Review

This page mirrors the real local skill used in the repository.

Use it after [Chapter 12](../chapter-12/README.md) whenever a ranking or recommendation system needs a system-level review instead of a scoreboard-only review.

Real skill source:

`.agents/skills/ml-ranking-review/SKILL.md`

Invoke in Codex:

`$ml-ranking-review`

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

- `System Objective`
- `What the Objective Leaves Out`
- `Metric Stack`
- `Feedback Loop Risks`
- `Exploration Sufficiency`
- `Offline vs Online Evidence`
- `Exposure / Fairness Concerns`
- `Recommendation`

## Try It Now

```text
Use $ml-ranking-review to review a recommendation feed system. Include the system objective, what that objective leaves out, the metric stack, feedback loop risks, whether exploration is sufficient, whether offline gains are likely to hold online, and a rollout recommendation.
```
