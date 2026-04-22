---
name: ml-neural-project
description: Turn a neural network experiment into a reproducible engineering project with clear configuration, split rules, module boundaries, logging, validation, checkpointing, and a practical debug checklist. Use when a learner or engineer has a neural task and wants a project structure another person can rerun, inspect, and extend.
---

# ML Neural Project

Use this skill when a neural experiment needs to become a reproducible project instead of a fragile one-off run.

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

Use this structure:

- `Task`
- `Data Source and Split Rules`
- `Config Boundary`
- `Model Module Boundaries`
- `Training Loop Logging`
- `Validation Procedure`
- `Checkpoint / Reproducibility Notes`
- `Debug Checklist`
- `Project Recommendation`

## Quality Bar

- Do not leave split logic implicit.
- Do not hide critical settings inside ad hoc notebook cells.
- Logging should support diagnosis, not vanity.
- Reproducibility notes should include seeds, versions, and checkpoint naming.

## Good Prompt Shapes

- Turn this text classification experiment into a reusable neural project plan.
- Review whether this PyTorch training setup is reproducible enough for a team.
- What project structure should we use for this neural ranking model?

## Reference

Read [references/project-readme-template.md](references/project-readme-template.md) for the experiment README template and setup checklist.
