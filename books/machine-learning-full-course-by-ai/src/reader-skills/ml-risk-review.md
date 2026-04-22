# Reader Skill: ML Risk Review

This page mirrors the real local skill used in the repository.

Use it after [Chapter 15](../chapter-15/README.md) whenever a system needs a concrete risk review before launch or after incidents.

Real skill source:

`.agents/skills/ml-risk-review/SKILL.md`

Invoke in Codex:

`$ml-risk-review`

## What This Skill Does

This skill identifies:

- main failure surfaces
- who is harmed by each failure
- which harms are likely, severe, or hidden
- existing mitigations
- remaining unacceptable risks
- monitoring and incident-review plans

## Workflow

1. Map the system workflow.
2. List likely and severe failure modes.
3. Identify who bears the cost.
4. Review whether human oversight is actually effective.
5. Define monitoring and escalation paths for risky failures.

## Output Format

- `System`
- `Main Failure Surfaces`
- `Who Is Harmed`
- `Likelihood / Severity / Hiddenness`
- `Existing Mitigations`
- `Human Oversight Reality Check`
- `Remaining Unacceptable Risks`
- `Monitoring / Escalation Plan`
- `Risk Verdict`

## Try It Now

```text
Use $ml-risk-review to review the risks of a support assistant that sometimes hallucinates policy answers. Identify main failure surfaces, who is harmed, which harms are likely or hidden, whether human review is actually effective, what risks remain unacceptable, and what monitoring signals should force redesign or rollback.
```
