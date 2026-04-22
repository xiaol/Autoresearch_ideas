# Chapter 5. Trees, Ensembles, and Strong Baselines

The machine learning world loves novelty.

Teams enjoy announcing new architectures, new modalities, and new performance records. But in many real tabular problems, one of the most embarrassing truths in applied ML is this:

strong tree-based baselines are often hard to beat.

That is not because innovation stopped.

It is because many real business datasets have a structure that trees and ensembles exploit extremely well:

- mixed feature types
- nonlinear thresholds
- missing values
- irregular interactions
- moderate-sized data

This chapter is about learning why decision trees and ensembles became such reliable workhorses, and why a professional ML engineer treats strong baselines as a form of honesty.

## 5.1 Case: Fraud Detection Where the Glamorous Model Loses

Imagine a payments company building a fraud model.

The feature table includes:

- transaction amount
- merchant category
- device age
- card country
- user history aggregates
- hour of day
- velocity features

The team is excited about deep learning. There are papers, embeddings, graph ideas, sequence ideas, and a sense that "serious ML" should look sophisticated.

But one awkward result keeps appearing:

- a well-tuned gradient boosting model keeps outperforming the flashier alternatives

This is a useful moment, not a disappointing one.

It teaches a core lesson:

model choice should follow data structure and operational need, not status.

## 5.2 Decision Trees: Splitting the Space

A **decision tree** makes predictions by repeatedly splitting the feature space.

For example:

- is transaction amount greater than 500
- is device age less than 3 days
- is merchant risk score above threshold

Each split narrows the region of the input space until the example lands in a leaf.

### Why trees feel natural

Trees fit the way many real decisions are described operationally:

- if amount is high and device is new, be suspicious
- if support-ticket length is short but contains certain words, route urgently

Even if the final production model is not rule-based, tree thinking connects easily to business intuition.

### Their hidden power

Trees capture:

- nonlinear relationships
- threshold effects
- feature interactions

without needing you to hand-code those interactions explicitly.

That is one reason they became dominant in many tabular settings.

## 5.3 The Weakness of Single Trees

A single decision tree is interpretable, but it can be unstable.

Small changes in the training data can produce a noticeably different tree.

It can also overfit by memorizing quirks in the training set.

This is where ensembles come in.

The key idea is simple:

- many weak or moderately strong models can become a stronger system when combined well

## 5.4 Bagging and Random Forests

**Bagging** means training multiple models on resampled versions of the data and averaging their predictions.

This reduces variance.

For trees, the most famous bagging-based method is the **random forest**.

Random forests help because:

- each tree sees a slightly different version of the problem
- the ensemble averages out some of the instability

In practical terms, random forests are often:

- robust
- easy to use
- strong baselines
- less fragile than single trees

They are especially useful when you want a serious tabular baseline without immediately optimizing every detail.

## 5.5 Boosting: Learn From Mistakes

If bagging says, "train many models in parallel and average them," boosting says something more sequential:

- train a model
- pay attention to what it gets wrong
- train the next model to focus more on those hard cases
- repeat

This family includes methods such as gradient boosting and modern implementations like XGBoost, LightGBM, and CatBoost.

### Why boosting is so strong

Boosting often performs extremely well on structured tabular data because it can build a rich predictive surface gradually, correcting earlier weakness step by step.

This is why, in many Kaggle competitions and many business settings, boosted trees become the model to beat.

### The larger lesson

Boosting teaches an important mindset:

- a good system often improves not by replacing everything, but by focusing repeatedly on the errors that remain

That idea will return later in error analysis and system design.

## 5.6 Why Tree-Based Models Dominate Tabular Work

Let us be explicit.

Tree-based models often shine in tabular data because they handle:

- heterogeneous features
- missingness reasonably well
- nonlinear effects
- interaction structure
- medium-data regimes

This does not mean deep learning is useless on tables. It means the burden of proof is real.

If a neural model cannot clearly beat a strong tree baseline in a given setting, then the question is not "How do we justify the neural model anyway?" The question is "Why are we not respecting the data?"

That is part of mature engineering culture.

## 5.7 Feature Importance: Helpful but Dangerous

One reason trees are attractive is that they appear interpretable.

People want to know:

- which features mattered most

Feature importance can help, but it is also easy to misuse.

### Why caution is needed

Feature importance scores can be misleading because:

- correlated features can share or steal importance
- different importance definitions tell different stories
- importance is not causality
- global importance may hide segment-specific effects

So if a stakeholder says, "This model proves feature X causes fraud," the correct response is:

- no, it does not

At best, the model says:

- this feature helped prediction under this data and modeling setup

That is much narrower, and much more honest.

## 5.8 Duplicate Rows, Leakage, and Quiet Failure

Tabular modeling creates a special danger: it is easy to get impressive metrics for bad reasons.

Three common culprits are:

- duplicate rows leaking across splits
- future information hidden inside engineered features
- group leakage where the same entity appears in train and test in a misleading way

A boosted tree can exploit these shortcuts very effectively. In that sense, strong models are unforgiving: they reward sloppy data handling by making bad practice look successful.

That is why strong baselines and strong audits must go together.

## 5.9 Model Comparison as a Professional Habit

By now the book has built enough foundation to state a principle more forcefully:

**Model comparison is not a side note. It is part of engineering integrity.**

For a tabular problem, a disciplined progression might look like:

1. Naive baseline
2. Linear or logistic baseline
3. Tree baseline
4. Random forest
5. Gradient boosting
6. More specialized models only if clearly justified

This sequence does two things:

- it improves performance
- it improves understanding

That second part matters just as much.

## 5.10 Harness Lab: Build a Tabular Model Comparison Harness

Here is a simple **Tabular Model Comparison Harness**.

### Purpose

Ensure that tabular ML work compares model families honestly and documents where each one wins or fails.

### Inputs

- task definition
- label definition
- split strategy
- feature table

### Required outputs

1. Naive baseline result
2. Linear baseline result
3. Tree baseline result
4. Ensemble result
5. Slice comparison
6. Leakage check notes
7. Recommendation with justification

### Minimal workflow

1. Freeze the split strategy.
2. Run a naive baseline.
3. Run an interpretable linear baseline if applicable.
4. Run a single tree or shallow tree baseline.
5. Run random forest or boosted tree.
6. Inspect improvements by slice, not only globally.
7. Document whether added complexity changed useful decisions.

### Evidence artifact

Create a comparison table with:

- model family
- metric
- calibration or threshold notes
- runtime or complexity cost
- known weaknesses

This table keeps discussion grounded when stakeholders or teammates are tempted by novelty alone.

## 5.11 Common Failure Modes

### Failure Mode 1. Deep Model Prestige

The team assumes the more modern architecture must be superior, regardless of data shape.

Fix:

- require strong tree-based baselines on structured tabular problems

### Failure Mode 2. Single-Metric Triumph

One number improves, but no one checks slice stability, leakage risk, or operational cost.

Fix:

- compare on multiple dimensions, not just one leaderboard score

### Failure Mode 3. Feature Importance Overclaim

The team treats predictive importance like causal proof.

Fix:

- explicitly separate prediction usefulness from causal interpretation

### Failure Mode 4. Hidden Leakage Success

The boosted model is "amazing" only because future or duplicated information slipped in.

Fix:

- audit provenance, timing, duplicates, and grouping before celebrating

### Failure Mode 5. Baseline Shame

The learner thinks using trees means they are doing something less advanced.

Fix:

- remember that respecting the data is more advanced than performing sophistication

## Chapter Summary

Decision trees and ensembles are not old tools to be tolerated on the way to modern ML. They are often the right tools for structured data. Trees capture thresholds and interactions naturally, random forests reduce instability, and boosting often sets the standard on tabular tasks. Strong baselines are not a beginner habit. They are a professional defense against self-deception.

## Use This Skill Now

After this chapter, open [Reader Skill: ML Tabular Model Review](../reader-skills/ml-tabular-model-review.md).

Use it to compare a linear model against a tree-based family on one tabular task. The important habit is not only choosing a winner. It is learning to justify that choice in terms of leakage risk, slice behavior, calibration, and operational cost.

## Extension Exercises

1. Train a single decision tree and inspect where it feels intuitive versus brittle.
2. Compare a linear baseline with a boosted tree on one tabular dataset.
3. Write down three ways feature importance can be misinterpreted.
4. Build a one-page tabular comparison table for a real or public dataset.

## Further Reading

- [References](../references.md)
- [Chapter 6. Evaluation, Error Analysis, and Experiment Design](../chapter-06/README.md)
