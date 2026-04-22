# Reader Skill: ML Problem Framer

This page mirrors the real local skill used in the repository.

Use it after [Chapter 3](../chapter-03/README.md) whenever a project name sounds impressive but the actual decision, label, and deployment context are still fuzzy.

Real skill source:

`.agents/skills/ml-problem-framer/SKILL.md`

Invoke in Codex:

`$ml-problem-framer`

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
4. Audit the label: delayed, proxy-based, human-judged, incomplete, or direct.
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

## Try It Now

```text
Use $ml-problem-framer to turn this request into a framing memo: "We want to predict which support tickets should be escalated to a senior engineer." Include the decision supported, prediction moment, label definition, leakage risks, cost asymmetries, and the first baseline worth trying.
```

## Quality Bar

- Do not jump into model families before clarifying the decision.
- Treat labels as products of a process, not automatic truth.
- Name proxy labels explicitly.
- Prefer uncomfortable clarity over vague elegance.

## Reference Checklist

Ask:

1. What decision changes because of this system?
2. What is the prediction moment?
3. What is the business target?
4. What is the model target?
5. How is the label created?
6. What is delayed, noisy, or proxy-based?
7. What features are unavailable at prediction time?
8. What are the main error costs?
9. What baseline should be attempted first?
