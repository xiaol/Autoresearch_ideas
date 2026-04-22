# Reader Skill: ML Adaptation Decision

This page mirrors the real local skill used in the repository.

Use it after [Chapter 10](../chapter-10/README.md) whenever a team must choose between prompting, retrieval, fine-tuning, or a smaller bespoke route.

Real skill source:

`.agents/skills/ml-adaptation-decision/SKILL.md`

Invoke in Codex:

`$ml-adaptation-decision`

## What This Skill Does

This skill decides:

- the recommended adaptation path
- why simpler options are or are not enough
- what evaluation should follow
- what operational risks are introduced
- what would justify escalating to a heavier method

## Workflow

1. Define the task and failure cost.
2. Test the lightest plausible strategy first.
3. Add retrieval if knowledge grounding is the main issue.
4. Consider fine-tuning only if behavior must change more deeply.
5. Re-evaluate cost, latency, maintenance, and risk.

## Output Format

- `Task`
- `Main Failure Cost`
- `Recommended Adaptation Path`
- `Why Simpler Options Are or Are Not Enough`
- `Grounding / Knowledge Needs`
- `Latency / Cost Tradeoff`
- `Operational Risks`
- `Evaluation Plan`
- `Exit Criteria for Escalation`

## Try It Now

```text
Use $ml-adaptation-decision to choose between prompting, retrieval, and fine-tuning for an enterprise support assistant. Consider domain specificity, available labeled data, latency budget, cost budget, grounding requirements, and what would justify escalating to a heavier method.
```
