# How to Use Reader Skills with This Book

This book is not designed to be read only as a sequence of explanations.

It is designed to be used with reusable reader skills.

The core loop is simple:

1. Read the chapter until you can state the main judgment in your own words.
2. Choose the matching reader skill from [Appendix B. Reader Skill Catalog](appendix-b.md).
3. Run that skill on the chapter case, the runnable example, or one of your own problems.
4. Compare the skill output with your own reasoning instead of accepting it blindly.
5. Save the result as a memo, checklist, or review note that you can reuse later.

That is how the book turns chapter reading into a repeatable learning system.

## If You Are Using Codex

Inside a Codex-style environment, the reader skills in this repository can be invoked directly with `$skill-name`.

The normal pattern is:

1. Open the chapter.
2. Open the corresponding in-book skill page.
3. Invoke the skill with the chapter case or with your own real task.
4. Ask the skill to produce a judgment artifact, not only a short answer.
5. Compare that artifact with your own reasoning and revise it.

For example:

```text
Use $ml-baseline-builder to review a delivery-time prediction task. Include the naive baseline, the first interpretable baseline, the shared metric, critical slices, and whether a more complex model is justified yet.
```

The source skill files live under `.agents/skills/` in the repository, so readers can inspect the actual reusable workflow behind each skill.

## If You Are Not Using Codex

You can still use the skills.

Each reader-skill page and each Appendix B entry includes a copyable "Try It Now" scaffold. Treat that scaffold as a structured prompt template.

The workflow is:

1. Copy the chapter's skill scaffold.
2. Replace the example task with your own case.
3. Ask for a structured output such as a memo, review sheet, experiment plan, or checklist.
4. Compare the result with your own analysis.
5. Keep the parts that improved your judgment and rewrite the rest in your own words.

The point is not the product name of the tool.

The point is learning to reuse a disciplined thinking pattern.

## What To Save After Each Skill Run

After each run, save a small artifact with:

- the original task or case
- the skill you used
- the output you accepted
- the output you rejected or revised
- the next experiment or follow-up question

This turns AI help into accumulated engineering judgment instead of disposable chat history.

## Best Inputs For Reader Skills

Reader skills work best when you give them one of the following:

- a concrete equation
- a small dataset description
- a model comparison result
- a failure symptom
- a deployment plan
- a risk scenario
- a portfolio or growth snapshot

Vague requests are still possible, but the strongest results usually come from real chapter artifacts, real metrics, or real project notes.

## Chapter-To-Skill Map

| Chapter | Best Reader Skill |
| --- | --- |
| Chapter 2 | `ml-math-translator` |
| Chapter 3 | `ml-problem-framer` |
| Chapter 4 | `ml-baseline-builder` |
| Chapter 5 | `ml-tabular-model-review` |
| Chapter 6 | `ml-evaluation-review` |
| Chapter 7 | `ml-training-diagnostics` |
| Chapter 8 | `ml-neural-project` |
| Chapter 9 | `ml-architecture-reader` |
| Chapter 10 | `ml-adaptation-decision` |
| Chapter 11 | `ml-retrieval-evaluation` |
| Chapter 12 | `ml-ranking-review` |
| Chapter 13 | `ml-data-readiness` |
| Chapter 14 | `ml-launch-readiness` |
| Chapter 15 | `ml-risk-review` |
| Chapter 16 | `ml-professional-growth` |

Chapter 1 is different: it explains why this entire skill-and-harness approach matters in the first place. Use it to choose your first skill and start the habit.

## A Strong Default Workflow

If you want one repeatable habit for the whole book, use this:

1. Read the chapter summary.
2. Run the chapter's extension exercise or a matching example from [Appendix C. Runnable Example Cases](examples.md).
3. Invoke the matching reader skill.
4. Write a one-page memo in your own words.
5. Save that memo beside your notes or project files.

That workflow is the practical form of the book's main claim:

learning should leave behind reusable systems, not only memories.
