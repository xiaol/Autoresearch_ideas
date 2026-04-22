---
name: ml-architecture-reader
description: Turn an ML architecture paper or model explainer into structured engineering understanding by identifying the original task pressure, the core mechanism change, claimed gains, tradeoffs, and when the architecture should or should not matter. Use when reading transformers, sequence models, retrieval architectures, or other model papers without wanting to drift into buzzword recognition.
---

# ML Architecture Reader

Use this skill when a model paper should become operational understanding rather than passive admiration.

## What This Skill Does

This skill extracts:

- the problem the architecture addresses
- the core mechanism change
- the claimed advantage
- the tradeoff or new cost
- one case where it should help
- one case where it may not matter

## Workflow

1. Restate the task pressure.
2. Identify what the older approach struggled with.
3. Identify the new mechanism.
4. Translate the mechanism into plain language.
5. Ask what evidence would prove the gain is real.

## Output Format

Use this structure:

- `Paper / Architecture`
- `Task Pressure`
- `Old Limitation`
- `Core Mechanism Change`
- `Claimed Gain`
- `Tradeoff or New Cost`
- `When It Should Help`
- `When It May Not Matter`
- `Would You Use It`

## Quality Bar

- Do not repeat paper slogans without naming the old limitation.
- Always translate the mechanism into plain language.
- Name at least one tradeoff or new cost.
- End with judgment, not only summary.

## Good Prompt Shapes

- Explain what attention changes compared with recurrent models.
- Read this architecture note and tell me whether the gain is likely to matter for my task.
- Turn this transformer paper into a one-page engineering note.

## Reference

Read [references/architecture-note-template.md](references/architecture-note-template.md) for the one-page reading template.
