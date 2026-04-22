# Appendix B. Reader Skill Catalog

This appendix maps the book's harness ideas to actual local skills included in this repository so readers can try them directly.

These are not only chapter metaphors. They are real local skills in this repository. A reader can invoke them in a Codex-style environment with `$skill-name`.

Because mdBook only renders pages inside the book source tree, this appendix links to mirrored in-book skill pages. The source-of-truth skill files still live under `.agents/skills/` in the repository.

If you want the step-by-step workflow first, begin with [How to Use Reader Skills with This Book](how-to-use-reader-skills.md).

## Available Local Skills

### 1. ML Math Translator

In-book page:

- [ML Math Translator](reader-skills/ml-math-translator.md)

Source skill in the repository:

`.agents/skills/ml-math-translator/SKILL.md`

Try it now:

```text
Use $ml-math-translator to explain w <- w - eta * grad_w L for a beginner. Define every symbol, annotate shapes, give a tiny numeric example, map it to NumPy-style code, and name one common mistake.
```

Best use cases:

- gradient descent updates
- cross-entropy loss
- matrix multiplications in linear models
- attention equations in transformers

Book references:

- [Chapter 1](chapter-01/README.md)
- [Chapter 2](chapter-02/README.md)

### 2. ML Problem Framer

In-book page:

- [ML Problem Framer](reader-skills/ml-problem-framer.md)

Source skill in the repository:

`.agents/skills/ml-problem-framer/SKILL.md`

Try it now:

```text
Use $ml-problem-framer to turn this request into a framing memo: "We want to predict which support tickets should be escalated to a senior engineer." Include the decision supported, prediction moment, label definition, leakage risks, cost asymmetries, and the first baseline worth trying.
```

Best use cases:

- churn prediction requests
- fraud detection ideas
- support triage systems
- ranking or recommendation proposals

Book references:

- [Chapter 3](chapter-03/README.md)

### 3. ML Baseline Builder

In-book page:

- [ML Baseline Builder](reader-skills/ml-baseline-builder.md)

Source skill in the repository:

`.agents/skills/ml-baseline-builder/SKILL.md`

Try it now:

```text
Use $ml-baseline-builder to design a first comparison plan for a delivery-time prediction task. Include the naive baseline, the first interpretable baseline, the shared comparison metric, critical slices, and one reason a more complex model may or may not be justified.
```

Best use cases:

- first model planning
- regression or classification baselines
- baseline justification memos
- early model-family decisions

Book references:

- [Chapter 4](chapter-04/README.md)

### 4. ML Tabular Model Review

In-book page:

- [ML Tabular Model Review](reader-skills/ml-tabular-model-review.md)

Source skill in the repository:

`.agents/skills/ml-tabular-model-review/SKILL.md`

Try it now:

```text
Use $ml-tabular-model-review to compare naive, linear, tree, and boosted-tree baselines for a fraud detection task. Include split strategy, slice comparison, leakage notes, calibration concerns, and a justified recommendation.
```

Best use cases:

- structured tabular ML benchmarks
- fraud and risk models
- boosted-tree versus linear tradeoffs
- slice-based model comparison

Book references:

- [Chapter 5](chapter-05/README.md)

### 5. ML Evaluation Review

In-book page:

- [ML Evaluation Review](reader-skills/ml-evaluation-review.md)

Source skill in the repository:

`.agents/skills/ml-evaluation-review/SKILL.md`

Try it now:

```text
Use $ml-evaluation-review to review this fraud model result: AUC 0.94, recall 0.61 at the current threshold, precision 0.12, and weak performance on new merchants. Tell me what the main metric hides, whether thresholding or calibration matters, which slices are critical, and whether the experiment supports the claim that the new model is better.
```

Best use cases:

- imbalanced classification results
- model comparison summaries
- thresholding decisions
- deployment readiness reviews

Book references:

- [Chapter 1](chapter-01/README.md)
- [Chapter 6](chapter-06/README.md)

### 6. ML Training Diagnostics

In-book page:

- [ML Training Diagnostics](reader-skills/ml-training-diagnostics.md)

Source skill in the repository:

`.agents/skills/ml-training-diagnostics/SKILL.md`

Try it now:

```text
Use $ml-training-diagnostics to diagnose this training run: training accuracy rises quickly, validation accuracy stays flat, and the loss oscillates after the fifth epoch. Rank the likely causes, propose low-cost diagnostics, and recommend the next experiment that preserves attribution.
```

Best use cases:

- oscillating loss
- early overfitting
- train-validation mismatch
- suspiciously good or bad runs

Book references:

- [Chapter 7](chapter-07/README.md)

### 7. ML Neural Project

In-book page:

- [ML Neural Project](reader-skills/ml-neural-project.md)

Source skill in the repository:

`.agents/skills/ml-neural-project/SKILL.md`

Try it now:

```text
Use $ml-neural-project to turn a support-ticket classifier experiment into a reproducible project plan. Include config boundaries, split rules, model module boundaries, logging, validation, checkpointing, and a practical debug checklist.
```

Best use cases:

- PyTorch or JAX experiment structure
- reproducibility planning
- team handoff for neural projects
- turning notebook work into projects

Book references:

- [Chapter 8](chapter-08/README.md)

### 8. ML Architecture Reader

In-book page:

- [ML Architecture Reader](reader-skills/ml-architecture-reader.md)

Source skill in the repository:

`.agents/skills/ml-architecture-reader/SKILL.md`

Try it now:

```text
Use $ml-architecture-reader to explain what attention changes compared with recurrent models. State the original task pressure, the mechanism change, the claimed gain, one tradeoff, when it should help, and when it may not matter.
```

Best use cases:

- reading transformer papers
- sequence-model comparisons
- architecture notes
- translating research claims into engineering judgment

Book references:

- [Chapter 9](chapter-09/README.md)

### 9. ML Adaptation Decision

In-book page:

- [ML Adaptation Decision](reader-skills/ml-adaptation-decision.md)

Source skill in the repository:

`.agents/skills/ml-adaptation-decision/SKILL.md`

Try it now:

```text
Use $ml-adaptation-decision to choose between prompting, retrieval, and fine-tuning for an enterprise support assistant. Consider domain specificity, available labeled data, latency budget, cost budget, grounding requirements, and what would justify escalating to a heavier method.
```

Best use cases:

- LLM product planning
- retrieval versus fine-tuning decisions
- cost and latency tradeoffs
- adaptation strategy reviews

Book references:

- [Chapter 10](chapter-10/README.md)

### 10. ML Retrieval Evaluation

In-book page:

- [ML Retrieval Evaluation](reader-skills/ml-retrieval-evaluation.md)

Source skill in the repository:

`.agents/skills/ml-retrieval-evaluation/SKILL.md`

Try it now:

```text
Use $ml-retrieval-evaluation to review a company knowledge search system. Define what relevance means, summarize practical retrieval metrics, show representative good and bad results, classify failure patterns, and decide whether the main issue is representation, indexing, or ranking.
```

Best use cases:

- semantic search quality review
- knowledge retrieval systems
- query-set evaluation
- diagnosis of embedding versus ranking issues

Book references:

- [Chapter 11](chapter-11/README.md)

### 11. ML Ranking Review

In-book page:

- [ML Ranking Review](reader-skills/ml-ranking-review.md)

Source skill in the repository:

`.agents/skills/ml-ranking-review/SKILL.md`

Try it now:

```text
Use $ml-ranking-review to review a recommendation feed system. Include the system objective, what that objective leaves out, the metric stack, feedback loop risks, whether exploration is sufficient, whether offline gains are likely to hold online, and a rollout recommendation.
```

Best use cases:

- recommender systems
- ranking models with online effects
- offline versus online evaluation
- exploration and feedback-loop review

Book references:

- [Chapter 12](chapter-12/README.md)

### 12. ML Data Readiness

In-book page:

- [ML Data Readiness](reader-skills/ml-data-readiness.md)

Source skill in the repository:

`.agents/skills/ml-data-readiness/SKILL.md`

Try it now:

```text
Use $ml-data-readiness to review a delivery-time prediction dataset built from orders, courier events, weather snapshots, and customer support logs. Check timestamp semantics, feature freshness, join risks, leakage, train-serve mismatches, and ownership of the critical features.
```

Best use cases:

- multi-table tabular datasets
- feature-store reviews
- leakage audits
- pre-training data checks

Book references:

- [Chapter 13](chapter-13/README.md)

### 13. ML Launch Readiness

In-book page:

- [ML Launch Readiness](reader-skills/ml-launch-readiness.md)

Source skill in the repository:

`.agents/skills/ml-launch-readiness/SKILL.md`

Try it now:

```text
Use $ml-launch-readiness to review whether a support-ticket triage model is ready for production rollout. We have strong offline metrics, an online scoring service, a draft monitoring plan, and a basic rollback path. Tell me the serving-mode implications, missing deployment gates, monitoring gaps, drift exposure, and whether we should launch, do a limited rollout, or delay.
```

Best use cases:

- first production launches
- staged rollouts
- cost and latency reviews
- monitoring and rollback planning

Book references:

- [Chapter 14](chapter-14/README.md)

### 14. ML Risk Review

In-book page:

- [ML Risk Review](reader-skills/ml-risk-review.md)

Source skill in the repository:

`.agents/skills/ml-risk-review/SKILL.md`

Try it now:

```text
Use $ml-risk-review to review the risks of a support assistant that sometimes hallucinates policy answers. Identify main failure surfaces, who is harmed, which harms are likely or hidden, whether human review is actually effective, what risks remain unacceptable, and what monitoring signals should force redesign or rollback.
```

Best use cases:

- safety review before launch
- incident follow-up
- human-in-the-loop workflow review
- residual-risk documentation

Book references:

- [Chapter 15](chapter-15/README.md)

### 15. ML Professional Growth

In-book page:

- [ML Professional Growth](reader-skills/ml-professional-growth.md)

Source skill in the repository:

`.agents/skills/ml-professional-growth/SKILL.md`

Try it now:

```text
Use $ml-professional-growth to review my current ML profile. My recent artifacts are a fraud model project, a retrieval demo, and a launch-readiness memo. Tell me my current strengths, current weaknesses, one specialization direction to explore next, one portfolio artifact to create, one communication habit to improve, and the next review checkpoint.
```

Best use cases:

- quarterly growth reviews
- portfolio planning
- specialization decisions
- turning projects into evidence of judgment

Book references:

- [Chapter 16](chapter-16/README.md)

## How Readers Should Use Them

The intended progression is:

1. Learn the concept in the chapter.
2. Open the in-book skill page in the HTML version.
3. Invoke the corresponding skill directly with `$skill-name` if you are using Codex.
4. Reuse the same structure on a new problem of your own.
5. Compare the skill output with your own reasoning.

For a fuller workflow, use [How to Use Reader Skills with This Book](how-to-use-reader-skills.md) together with this appendix.

If a reader is using this repository inside Codex, the fifteen skills above are ready to use as real local skills. If a reader is using another environment, the in-book pages and the `SKILL.md` files still serve as reusable harness specifications and prompt scaffolds.

This is the book's central claim made practical: learning should leave behind reusable working systems, not only notes.
