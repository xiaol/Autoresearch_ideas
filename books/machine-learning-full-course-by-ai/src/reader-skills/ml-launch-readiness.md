# Reader Skill: ML Launch Readiness

This page mirrors the real local skill used in the repository.

Use it after [Chapter 14](../chapter-14/README.md) whenever a promising model is about to affect real users, agents, or business workflows.

Real skill source:

`.agents/skills/ml-launch-readiness/SKILL.md`

Invoke in Codex:

`$ml-launch-readiness`

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

## Try It Now

```text
Use $ml-launch-readiness to review whether a support-ticket triage model is ready for production rollout. We have strong offline metrics, an online scoring service, a draft monitoring plan, and a basic rollback path. Tell me the serving-mode implications, missing deployment gates, monitoring gaps, drift exposure, and whether we should launch, do a limited rollout, or delay.
```

## Quality Bar

- Do not treat offline metrics as launch proof.
- Monitoring is part of launch scope.
- Rollback must be explicit, not implied.
- State what would trigger a rollback or limited rollout.

## Reference Checklist

Ask:

1. Is the task batch or online?
2. What are the real latency and cost constraints?
3. What will be monitored after launch?
4. How will drift be detected?
5. What is the rollback or fallback path?
6. What deployment gates are required before wider rollout?
