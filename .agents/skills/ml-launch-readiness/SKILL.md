---
name: ml-launch-readiness
description: Review whether a machine learning system is ready for production rollout by checking serving mode, deployment gates, monitoring, latency, cost, rollback, drift exposure, and operational fallback plans. Use when a team is preparing to launch or expand an ML system beyond development.
---

# ML Launch Readiness

Use this skill when a model is no longer only a training result and is about to affect real users or workflows.

## What This Skill Does

This skill reviews production readiness across:

- batch versus online serving
- rollout gates
- monitoring
- latency and cost
- rollback and fallback
- drift exposure

## Workflow

1. Restate the operational stakes.
2. Decide what serving mode the task requires.
3. Check whether observability is sufficient.
4. Review rollout gates and rollout size.
5. Check rollback and fallback options.
6. State whether the system is ready, risky, or not ready.

## Output Format

Use this structure:

- `System`
- `Serving Mode`
- `Operational Stakes`
- `Monitoring Coverage`
- `Latency / Cost Risks`
- `Drift Exposure`
- `Rollback / Fallback`
- `Deployment Gates`
- `Launch Verdict`
- `Required Before Rollout`

## Quality Bar

- Do not treat offline metrics as launch proof.
- Monitoring is part of launch scope.
- Rollback must be explicit, not implied.
- State what would trigger a rollback or limited rollout.

## Good Prompt Shapes

- Is this model ready to launch?
- Review this deployment plan for production risks.
- What is missing from this rollout checklist?

## Reference

Read [references/launch-checklist.md](references/launch-checklist.md) for the structured launch review checklist.
