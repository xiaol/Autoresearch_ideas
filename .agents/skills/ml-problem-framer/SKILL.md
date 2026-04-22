---
name: ml-problem-framer
description: Turn a vague machine learning request into a decision-aware task definition with prediction moment, business target, label definition, leakage risks, cost asymmetries, and a first-baseline recommendation. Use when a project request sounds underspecified, label quality is unclear, or the real decision problem may differ from the project title.
---

# ML Problem Framer

Use this skill when someone says they want a churn model, fraud model, ranking model, or other ML system, but the actual decision, label, and deployment context are still fuzzy.

## What This Skill Does

This skill converts an ML request into a framing memo that answers:

- what decision is being supported
- what the prediction moment is
- what the business target is
- what the model target is
- how the label is produced
- where leakage or proxy-label risk may exist
- what baseline should be tried first

## Workflow

1. Rewrite the request as a decision problem.
2. Define the prediction timestamp or decision moment.
3. Separate business target from prediction target.
4. Audit the label:
   delayed, proxy-based, human-judged, incomplete, or direct.
5. List the biggest leakage risks.
6. Name false-positive and false-negative costs.
7. Recommend the simplest baseline worth trying.

## Output Format

Use this structure:

- `Request`
- `Decision Supported`
- `Prediction Moment`
- `Business Target`
- `Prediction Target`
- `Label Definition`
- `Label Risks`
- `Leakage Risks`
- `Cost Asymmetries`
- `First Baseline`
- `Open Questions`

## Quality Bar

- Do not jump into model families before clarifying the decision.
- Treat labels as products of a process, not automatic truth.
- Name proxy labels explicitly.
- Prefer uncomfortable clarity over vague elegance.

## Good Prompt Shapes

- Frame a churn prediction request for a SaaS company.
- Review whether this fraud detection task is actually well-defined.
- Turn this product brief into an ML framing memo.

## Reference

Read [references/framing-memo-template.md](references/framing-memo-template.md) for the exact memo template and the audit checklist.
