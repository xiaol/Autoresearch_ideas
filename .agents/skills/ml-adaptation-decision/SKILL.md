---
name: ml-adaptation-decision
description: Choose between prompting, retrieval, fine-tuning, parameter-efficient adaptation, or a smaller bespoke model using explicit task, cost, latency, grounding, and maintenance criteria. Use when a team is deciding how to adapt a pretrained or foundation model and wants an evidence-based recommendation instead of tool-chasing.
---

# ML Adaptation Decision

Use this skill when a team must choose how to adapt an existing model without drifting into expensive reflexes.

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

Use this structure:

- `Task`
- `Main Failure Cost`
- `Recommended Adaptation Path`
- `Why Simpler Options Are or Are Not Enough`
- `Grounding / Knowledge Needs`
- `Latency / Cost Tradeoff`
- `Operational Risks`
- `Evaluation Plan`
- `Exit Criteria for Escalation`

## Quality Bar

- Start with the lightest plausible path.
- Separate knowledge problems from behavior problems.
- Treat retrieval as a first-class option when freshness or proprietary knowledge matters.
- Include maintenance burden in the decision, not only model quality.

## Good Prompt Shapes

- Should we use prompting, retrieval, or fine-tuning for this support assistant?
- Review this adaptation plan for a domain-specific LLM task.
- What would justify moving from prompting to PEFT here?

## Reference

Read [references/adaptation-memo-template.md](references/adaptation-memo-template.md) for the structured adaptation memo.
