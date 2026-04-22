# Reader Skill: ML Neural Project

This page mirrors the real local skill used in the repository.

Use it after [Chapter 8](../chapter-08/README.md) whenever a neural experiment should become reproducible, debuggable, and team-readable.

Real skill source:

`.agents/skills/ml-neural-project/SKILL.md`

Invoke in Codex:

`$ml-neural-project`

## What This Skill Does

This skill helps specify:

- configuration boundaries
- data split rules
- model module boundaries
- training loop logging
- validation procedure
- checkpointing and reproducibility notes
- an ordinary failure checklist

## Workflow

1. Freeze the task and main metric.
2. Separate configuration from code.
3. Make input-output shapes explicit.
4. Specify logging that helps interpret training.
5. Record seeds, versions, and checkpoint naming.
6. Add a short debug checklist before scaling experiments.

## Output Format

- `Task`
- `Data Source and Split Rules`
- `Config Boundary`
- `Model Module Boundaries`
- `Training Loop Logging`
- `Validation Procedure`
- `Checkpoint / Reproducibility Notes`
- `Debug Checklist`
- `Project Recommendation`

## Try It Now

```text
Use $ml-neural-project to turn a support-ticket classifier experiment into a reproducible project plan. Include config boundaries, split rules, model module boundaries, logging, validation, checkpointing, and a practical debug checklist.
```
