# Chapter 6. Evaluation, Error Analysis, and Experiment Design

This chapter may be the most important in the entire book.

Why?

Because many people can train models.

Far fewer people can tell whether the model is actually good, good for whom, good under what conditions, and good enough to trust.

That difference is the beginning of engineering judgment.

In this chapter we return to two familiar cases.

First, a delivery-time model looks strong on average error, but it becomes unreliable exactly when customers care most: rain, rush hour, and long-distance routes.

Second, a fraud model boasts excellent overall accuracy, yet floods investigators with false positives and slows legitimate transactions.

Both models look respectable until evaluation becomes honest.

## 6.1 When a Good Metric Lies

Suppose your delivery model reports a low average error.

The team celebrates.

But later you inspect specific conditions and discover:

- performance is much worse during heavy rain
- performance is worse for certain neighborhoods
- performance is worse for high-order-volume restaurants

Now ask:

Was the model ever truly good?

The answer is:

- only under a weak definition of good

This is the central lesson of evaluation:

metrics are not self-interpreting.

They must be read in the context of:

- the decision
- the cost of errors
- the user experience
- the slices that matter

## 6.2 Accuracy Is Not Enough

Consider fraud detection.

Fraud is usually rare.

If only 1 percent of transactions are fraudulent, then a model that predicts "not fraud" for everything can be 99 percent accurate.

That is a useless model.

This is why evaluation is inseparable from class balance and error cost.

If a fraud model catches more fraud but also triples the false-positive workload for investigators, then operationally it may be worse even if a headline metric improves.

This is the first habit to build:

always ask what a metric hides.

## 6.3 Metrics as Tradeoffs, Not Decorations

Metrics are not badges. They are compressed summaries of tradeoffs.

### Regression metrics

For delivery-time prediction, common choices include:

- mean absolute error
- mean squared error
- root mean squared error

Each tells a slightly different story.

For example:

- mean absolute error is easier to interpret directly
- mean squared error punishes large mistakes more heavily

If customers are especially sensitive to very late deliveries, a metric that reacts more strongly to large misses may better reflect operational pain.

### Classification metrics

For fraud, support routing, or churn, common metrics include:

- precision
- recall
- F1
- ROC-AUC
- PR-AUC

Each of these emphasizes something different.

If investigator time is scarce, precision matters a lot.

If missing fraud is extremely expensive, recall matters a lot.

There is no universal "best metric." There is only the metric that reflects the decision context best.

## 6.4 Precision, Recall, and Thresholds

Let us slow down on a key point.

Many classifiers produce a score, not a final decision. A threshold converts score into action.

That means model behavior is shaped not only by training, but by threshold choice.

### Precision

Of the cases the model flagged as positive, how many were truly positive?

### Recall

Of the truly positive cases, how many did the model catch?

These are often in tension.

Raise the threshold:

- precision may improve
- recall may drop

Lower the threshold:

- recall may improve
- precision may fall

This is not a mathematical inconvenience. It is an operational design choice.

The right threshold depends on:

- cost of false positives
- cost of false negatives
- human review capacity
- downstream workflow

## 6.5 Calibration: Can We Trust the Probabilities?

A model can rank cases well and still be badly calibrated.

Calibration asks:

- when the model says 0.8, does that really correspond to something like 80 percent likelihood

This matters because many systems use scores for prioritization, not just hard classification.

In fraud:

- poorly calibrated scores can distort investigator queues

In support routing:

- poorly calibrated urgency scores can waste escalations

In churn:

- poorly calibrated risk scores can misallocate retention effort

Calibration is one of the places where professional ML begins to look different from classroom ML. It asks not only whether the ordering is useful, but whether the confidence can support action.

## 6.6 Train, Validation, and Test: Discipline Over Convenience

One of the easiest ways to lie to yourself is to let the evaluation boundary become fuzzy.

That is why we separate:

- training data
- validation data
- test data

### Training data

Used to fit parameters.

### Validation data

Used to choose models, thresholds, or hyperparameters.

### Test data

Used sparingly to estimate final generalization after the important choices are already made.

The purpose of this separation is not ritual purity. It is to preserve honesty.

Every time you look at a split and adjust the system based on what you saw, that split is becoming less like a clean final estimate and more like part of development.

This is why evaluation discipline is really about resisting convenience.

## 6.7 Slice-Based Evaluation

Global averages are often where important failures hide.

That is why slice-based evaluation is essential.

A slice may be defined by:

- geography
- device type
- merchant category
- user segment
- time of day
- weather condition
- ticket language

For the delivery case, slice analysis might reveal:

- urban core works well
- suburban long-distance routes fail badly

For fraud:

- one payment method is over-flagged
- one region sees much lower recall

Slices matter because deployment happens to actual groups, not to the average row in a spreadsheet.

## 6.8 Error Analysis: Study the Failures, Not Just the Score

After evaluating metrics, the next step is **error analysis**.

Error analysis asks:

- where does the model fail
- what patterns define the failures
- what hypotheses do those failures suggest

For delivery-time prediction, error analysis might show:

- large underestimates mostly happen on rainy evenings
- large overestimates occur for small orders near fast restaurants

For fraud:

- false positives concentrate on new devices in certain merchant segments
- false negatives spike in low-amount coordinated attacks

That kind of information is far more useful than a single number.

It tells you:

- which features may be missing
- which slices need attention
- whether the label process is suspect
- whether the model family is mismatched

## 6.9 Experiment Design: Change One Thing, Learn One Thing

Many experiment logs are noisy because the team changes too many things at once.

Then when the result moves, no one knows why.

Good experiment design tries to preserve attribution.

That means:

- define the hypothesis
- change one important variable at a time when possible
- state what success would look like
- record the outcome
- record what the outcome actually means

This habit matters because the job of an experiment is not merely to improve the metric. The job is to produce knowledge that the team can reuse.

## 6.10 An Experiment Log That Deserves to Exist

Many teams keep logs that no one trusts or rereads.

Let us define a better one.

A good experiment log should include:

- date
- task
- split version
- hypothesis
- exact change made
- primary metric
- slice effects
- calibration or threshold notes
- result
- interpretation
- next action

Notice the word interpretation.

An experiment without interpretation is only a number in a table.

## 6.11 Harness Lab: Build an Evaluation Review Harness

This chapter needs a harness more than almost any other chapter.

Here is a simple **Evaluation Review Harness**.

### Purpose

Pressure-test whether a model result is trustworthy enough to inform the next decision.

### Inputs

- task definition
- chosen metrics
- split strategy
- slice results
- calibration or threshold settings
- recent experiment logs

### Required outputs

1. Whether the main metric is appropriate
2. What the metric hides
3. Which slices are most concerning
4. Whether calibration matters here
5. Whether thresholding is operationally sane
6. Whether the experiment design supports the claimed conclusion
7. Recommended next step

### Minimal workflow

1. Restate the business target.
2. Check whether the primary metric reflects it.
3. Inspect class balance or error distribution.
4. Review slice-level behavior.
5. Review thresholding and confidence quality.
6. Examine recent experimental attribution.
7. Decide whether to deploy, debug, or redesign.

### Evidence artifact

Produce a one-page evaluation review for each serious model candidate.

It should answer:

- what looks good
- what looks weak
- what we still do not know
- what must be true before rollout is justified

This artifact is one of the clearest markers of engineering maturity in an ML team.

## 6.12 Common Failure Modes

### Failure Mode 1. Average Score Worship

The global metric looks good, so the team ignores the slices where the product actually hurts.

Fix:

- require slice evaluation before any serious conclusion

### Failure Mode 2. Accuracy Illusion

The team celebrates high accuracy on an imbalanced task.

Fix:

- choose metrics aligned with class imbalance and operational cost

### Failure Mode 3. Threshold Amnesia

The team talks as if the classifier has one fixed behavior, forgetting that score-to-action is a design choice.

Fix:

- treat threshold selection as part of deployment design

### Failure Mode 4. Test-Set Leakage by Curiosity

The team keeps checking the test set and quietly tuning against it.

Fix:

- protect the evaluation boundary and document when a split stops being clean

### Failure Mode 5. Experiment Fog

Multiple changes happen at once, so nobody knows what actually improved.

Fix:

- log hypotheses and preserve attribution discipline

## Chapter Summary

Machine learning becomes engineering when evaluation becomes honest. Metrics summarize tradeoffs, thresholds create operational behavior, calibration determines whether scores can be trusted, and slice analysis reveals where average performance hides real damage. Error analysis turns failure into diagnosis, and good experiment design turns metric changes into reusable knowledge. A strong practitioner does not ask only, "Did the score improve?" They ask, "Can I trust what this result means?"

## Use This Skill Now

After this chapter, open [Reader Skill: ML Evaluation Review](../reader-skills/ml-evaluation-review.md).

Feed it one metric summary from your own project or one invented from this chapter. Then check whether the skill exposed hidden threshold, calibration, or slice issues that your first reading missed.

## Extension Exercises

1. Take one classification problem and explain why accuracy alone could be misleading.
2. For a model with probabilistic output, describe how changing the threshold changes the workflow.
3. Draft a slice-based evaluation plan for either delivery-time prediction or fraud detection.
4. Create an experiment log template and fill it with one hypothetical experiment.

## Further Reading

- [References](../references.md)
- [Chapter 7. Optimization and Representation Learning](../chapter-07/README.md)
