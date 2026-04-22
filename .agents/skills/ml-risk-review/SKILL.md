---
name: ml-risk-review
description: Review a machine learning or AI system for concrete safety and responsibility risks by identifying failure surfaces, harmed groups, severity, hidden exposure, mitigation gaps, and escalation or monitoring plans. Use when a system is approaching launch, has suffered incidents, or needs a realistic risk review rather than high-level principles alone.
---

# ML Risk Review

Use this skill when safety should be treated as ordinary engineering review rather than abstract posture.

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

Use this structure:

- `System`
- `Main Failure Surfaces`
- `Who Is Harmed`
- `Likelihood / Severity / Hiddenness`
- `Existing Mitigations`
- `Human Oversight Reality Check`
- `Remaining Unacceptable Risks`
- `Monitoring / Escalation Plan`
- `Risk Verdict`

## Quality Bar

- Connect every concern to a real workflow or failure mode.
- Do not assume human review is effective without examining the actual review design.
- Benchmarks alone are not safety proof.
- Name what signal would force redesign or rollback.

## Good Prompt Shapes

- Review the risks of this support assistant before launch.
- What harms remain exposed in this human-in-the-loop system?
- Turn this safety discussion into a concrete risk memo.

## Reference

Read [references/risk-memo-template.md](references/risk-memo-template.md) for the risk review memo structure.
