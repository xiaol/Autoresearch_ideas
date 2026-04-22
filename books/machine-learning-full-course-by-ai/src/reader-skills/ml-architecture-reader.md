# Reader Skill: ML Architecture Reader

This page mirrors the real local skill used in the repository.

Use it after [Chapter 9](../chapter-09/README.md) whenever a paper or architecture explainer feels impressive but not yet operational.

Real skill source:

`.agents/skills/ml-architecture-reader/SKILL.md`

Invoke in Codex:

`$ml-architecture-reader`

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

- `Paper / Architecture`
- `Task Pressure`
- `Old Limitation`
- `Core Mechanism Change`
- `Claimed Gain`
- `Tradeoff or New Cost`
- `When It Should Help`
- `When It May Not Matter`
- `Would You Use It`

## Try It Now

```text
Use $ml-architecture-reader to explain what attention changes compared with recurrent models. State the original task pressure, the mechanism change, the claimed gain, one tradeoff, when it should help, and when it may not matter.
```
