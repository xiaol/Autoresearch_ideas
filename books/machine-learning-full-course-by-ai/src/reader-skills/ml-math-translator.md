# Reader Skill: ML Math Translator

This page mirrors the real local skill used in the repository.

Use it after [Chapter 2](../chapter-02/README.md) whenever notation feels clear for a moment but does not yet feel operational.

Real skill source:

`.agents/skills/ml-math-translator/SKILL.md`

Invoke in Codex:

`$ml-math-translator`

## What This Skill Does

For any formula, derivation, or notation block, the skill produces:

1. Symbol meanings
2. Shape annotations when relevant
3. A plain-English explanation
4. A small concrete example
5. A code mapping when useful
6. One likely misconception or edge case

This is especially useful for:

- gradient descent updates
- loss functions
- matrix-vector expressions
- probability notation
- transformer equations

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

Skip `Shapes` or `Code Mapping` only when they truly do not apply.

## Try It Now

```text
Use $ml-math-translator to explain w <- w - eta * grad_w L for a beginner. Define every symbol, annotate shapes, give a tiny numeric example, map it to NumPy-style code, and name one common mistake.
```

## Quality Bar

- Do not hide behind jargon.
- Do not explain intuition without grounding it in the symbols.
- Do not explain symbols without connecting them to an operation.
- Prefer one clean example over many shallow ones.

## Reference Checklist

For each formula:

1. Copy the formula exactly.
2. Define each symbol.
3. State tensor, vector, or matrix shapes if relevant.
4. Explain the operation in one plain-English sentence.
5. Give one small example or code fragment.
6. Name one misconception or edge case.
