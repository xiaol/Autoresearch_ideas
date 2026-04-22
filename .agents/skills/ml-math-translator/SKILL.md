---
name: ml-math-translator
description: Translate machine learning math, notation, and formulas into plain language, symbol definitions, shape annotations, code mappings, and likely misconceptions. Use when a learner or engineer needs help understanding equations, gradients, losses, matrix expressions, or derivations without hand-wavy explanation.
---

# ML Math Translator

Use this skill when the task is to make ML notation operational rather than intimidating.

## What This Skill Does

For any formula, derivation, or notation block, produce:

1. Symbol meanings
2. Shape annotations when relevant
3. Plain-English explanation
4. A small concrete example
5. A code mapping when useful
6. One likely misconception or edge case

This skill is especially good for:

- gradient descent updates
- loss functions
- matrix-vector expressions
- probability notation
- transformer or embedding equations

## Workflow

1. Restate the exact formula.
2. Name every symbol before explaining the whole expression.
3. Annotate shapes if tensors, vectors, or matrices are involved.
4. Explain the operation in one plain-English sentence.
5. Give a small worked example or code mapping.
6. Add one common misunderstanding.

## Output Format

Use this structure:

- `Formula`
- `Symbols`
- `Shapes`
- `Plain English`
- `Tiny Example`
- `Code Mapping`
- `Common Mistake`

Skip `Shapes` or `Code Mapping` only if they do not apply.

## Quality Bar

- Do not hide behind jargon.
- Do not give only intuition without grounding it in the actual symbols.
- Do not give only the symbolic explanation without operational meaning.
- Prefer one clean example over many shallow ones.

## Good Prompt Shapes

- Explain `w <- w - eta * grad(L)` for a beginner and annotate shapes.
- Translate the cross-entropy loss into plain English and code.
- Walk through `Xw + b` and explain what each object means.

## Reference

Read [references/checklist.md](references/checklist.md) when you want the exact translation checklist and a reusable response template.
