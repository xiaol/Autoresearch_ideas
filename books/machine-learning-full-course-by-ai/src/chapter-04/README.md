# Chapter 4. First Models: Linear Models and Nearest Neighbors

The first serious models you learn matter more than many people realize.

Not because they will always be the best production models.

Not because the industry has stopped moving.

But because these models teach some of the most durable ideas in machine learning:

- what a feature really is
- what a baseline is supposed to do
- how a decision boundary emerges
- how scaling changes behavior
- why interpretability and discipline often beat premature complexity

In this chapter we will use two cases.

The first is our recurring delivery-time prediction problem, where linear regression gives us a natural first model.

The second is support-ticket urgency classification, where a nearest-neighbor baseline helps us think about similarity before we rush into heavier machinery.

## 4.1 Why First Models Matter

Beginners often assume that simple models are stepping stones to be discarded quickly. That is the wrong mindset.

Simple models are not only educational. They are diagnostic.

A good first model helps you answer questions like:

- are the features informative at all
- is the target stable enough to learn
- is the task linearly structured or obviously nonlinear
- does the data pipeline make sense
- does a complicated model have any reason to exist

The habit you want to build is this:

Before reaching for sophistication, earn the right to need it.

That is what baselines are for.

## 4.2 Case One: Delivery-Time Prediction with Linear Regression

Return to the logistics platform from Chapter 2.

We want to predict delivery time using features like:

- distance
- item count
- weather
- traffic
- restaurant load

This is a regression problem because the target is a number.

The natural first baseline is **linear regression**.

### The basic idea

Linear regression assumes the prediction can be written as a weighted combination of the features:

$$
\hat{y} = x \cdot w + b
$$

This is powerful for two reasons.

First, it is easy to understand.

Second, when it fails, the pattern of failure often teaches you something important:

- the relationship may be nonlinear
- the features may be badly scaled
- critical interactions may be missing
- the target may be noisier than expected

### What linear regression teaches

It teaches that a model is a structured argument about relevance.

Each weight says:

- this feature matters more
- this feature matters less
- this feature may matter in the opposite direction

That is why linear models are pedagogically valuable. They make the model's structure inspectable.

### A healthy expectation

Linear regression will not capture everything. Delivery time may rise nonlinearly under traffic spikes or restaurant overload.

That is fine.

The goal of the first model is not perfection. The goal is clarity:

- can we get a sensible baseline
- do the signs of the weights make intuitive sense
- where does the model break

## 4.3 Regression Versus Classification

Before we move on, it is worth drawing a clean distinction.

### Regression

Predict a continuous value.

Examples:

- delivery time
- house price
- demand forecast

### Classification

Predict a category or class.

Examples:

- urgent versus not urgent
- fraud versus legitimate
- churn versus retain

This sounds obvious, but the distinction matters because the model output, loss function, and evaluation metrics all change with the task.

In regression, you care about how far off the number is.

In classification, you care about which class is chosen, how confident the model is, and what the costs of different mistakes are.

## 4.4 Case Two: Support-Ticket Urgency with Logistic Regression

Now imagine a support platform that receives thousands of tickets per day.

The business wants to route:

- urgent tickets quickly
- ordinary tickets normally

That is a classification problem.

One good first model is **logistic regression**.

### Why logistic regression matters

Logistic regression is one of the most useful models in ML education because it sits at a productive boundary:

- still simple enough to understand
- still strong enough to be useful
- naturally connected to probability
- often competitive on real data when features are reasonable

It predicts a score that can be interpreted as probability after the logistic transformation:

$$
P(y=1 \mid x) = \sigma(x \cdot w + b)
$$

where \(\sigma\) is the sigmoid function.

Plain English:

- compute a weighted score
- squash it into a value between 0 and 1
- interpret that value as class probability

### Why this is helpful

Now the model is not only giving a class label. It is giving a probability-like score.

That lets us reason about:

- thresholds
- confidence
- calibration
- asymmetric costs

If urgent tickets are rare, we may not want to classify based on a naive threshold of 0.5. That decision should depend on operational cost.

This is one reason logistic regression is such a good bridge model. It connects feature-based thinking to decision-making.

## 4.5 Nearest Neighbors: Similarity as a Model

Not every baseline has to learn weights.

Sometimes the question is simpler:

- which past examples look most like the current one

That is the core idea behind **nearest neighbors**.

For support-ticket urgency, a nearest-neighbor baseline might say:

- find the tickets with text or metadata most similar to this new ticket
- inspect how those were labeled
- use that neighborhood to classify the new one

### Why nearest neighbors matters

It teaches an entirely different intuition from linear models.

Linear models say:

- combine features globally using learned weights

Nearest neighbors says:

- classify locally by similarity

This is a very important conceptual distinction. Some tasks are better understood through local structure than through one global linear rule.

### The hidden lesson

Nearest neighbors also forces you to think hard about:

- distance metrics
- feature scaling
- representation quality

If one feature is measured in thousands and another in tiny decimals, the larger-scale feature can dominate distance unless you scale the inputs carefully.

That is why nearest neighbors is both simple and educational. It exposes assumptions that other models can hide.

## 4.6 Feature Scaling and Geometry

Feature scaling sounds like a preprocessing detail. It is actually part of the geometry of the problem.

Suppose we use:

- distance in kilometers
- item count
- traffic score
- support-ticket word count

If one feature ranges from 0 to 10,000 and another from 0 to 1, then raw comparisons can become distorted.

This matters especially for:

- nearest neighbors
- logistic regression with regularization
- gradient-based optimization

Scaling helps ensure that features participate more comparably in optimization or distance calculations.

The broader lesson is useful:

Model behavior is often shaped not only by architecture, but by representation and scale.

## 4.7 Regularization: Prefer Simpler Explanations When Possible

Once a model can fit training data, another question appears:

- how hard should we let it fit

**Regularization** is one family of answers.

In plain language, regularization tells the model:

- explain the data, but do not become unnecessarily wild

For linear models, regularization often means penalizing large weights.

That can help when:

- features are noisy
- features are correlated
- the model is overreacting to quirks in the training set

This matters because overfitting does not begin only with deep networks. Even simple models can become brittle when the data is messy or the feature space is badly engineered.

## 4.8 Baseline Discipline

Let us state one of the strongest habits in practical ML:

**Always know what your new idea is beating.**

This sounds trivial. It is not.

A weak baseline can make mediocre progress look impressive.

A strong baseline can save a team months of wasted effort.

For delivery-time prediction, your baseline progression might be:

1. Predict the average delivery time.
2. Predict by simple grouped averages.
3. Linear regression.
4. Add regularization and better features.

For support-ticket urgency:

1. Majority class prediction.
2. Keyword heuristic.
3. Logistic regression.
4. Nearest-neighbor comparison.

This layering matters because it turns improvement into evidence rather than theater.

## 4.9 From Simple Models to Better Questions

An underrated benefit of first models is that they sharpen the next question.

After a linear regression baseline, you may ask:

- are interactions missing
- are features nonlinear
- are there systematic slice errors

After logistic regression, you may ask:

- are the classes separable with current features
- are we thresholding well
- do we need richer representations

After nearest neighbors, you may ask:

- is the distance measure wrong
- do we need embeddings
- are the raw features too crude

This is what strong baselines do. They generate better next steps.

## 4.10 Harness Lab: Build a Baseline Comparison Harness

We now turn the chapter into a reusable workflow.

Here is a simple **Baseline Comparison Harness**.

### Purpose

Ensure that every new modeling effort begins with interpretable baselines and explicit comparison rules.

### Inputs

- task type
- target definition
- available features
- operational constraints

### Required outputs

1. Simplest naive baseline
2. First interpretable model
3. Metric to compare on
4. Slice checks
5. One reason a more complex model may be justified
6. One reason it may not be justified

### Minimal workflow

1. Define the task clearly.
2. Choose a naive baseline.
3. Choose a first interpretable model.
4. Compare them on the same split.
5. Inspect where each fails.
6. Only then consider richer models.

### Evidence artifact

Produce a one-page comparison memo:

- task
- baselines tried
- metrics
- failure patterns
- recommendation for the next model class

This memo prevents complexity from arriving as a reflex.

## 4.11 Common Failure Modes

### Failure Mode 1. Skipping the Baseline

The team jumps to a complex model before learning whether simple structure already solves much of the problem.

Fix:

- require at least one naive baseline and one interpretable baseline

### Failure Mode 2. Confusing Interpretability with Weakness

The learner assumes a simpler model is only for beginners.

Fix:

- treat simple models as instruments for understanding the task

### Failure Mode 3. Ignoring Scale

Nearest neighbors or regularized models behave strangely because the features are on incompatible scales.

Fix:

- inspect feature ranges and normalize when appropriate

### Failure Mode 4. Using the Wrong Loss for the Task

The model may be technically implemented but conceptually mismatched to regression or classification.

Fix:

- restate the task type and output before selecting the model

### Failure Mode 5. Believing a Better Metric Without Understanding Why

The score improved, but the team cannot explain what changed structurally.

Fix:

- tie every improvement to either features, representation, thresholding, or model capacity

## Chapter Summary

Linear regression, logistic regression, and nearest neighbors are not disposable beginner tools. They teach the geometry of machine learning, the meaning of similarity, the importance of scaling, and the discipline of baselines. A first model should make the task more legible, not merely produce a score. The goal is not to stay simple forever. The goal is to make every increase in complexity feel earned.

## Use This Skill Now

After this chapter, open [Reader Skill: ML Baseline Builder](../reader-skills/ml-baseline-builder.md).

Run the delivery-time example from [Appendix C. Runnable Example Cases](../examples.md) or use one of your own tasks. Ask the skill for the naive baseline, the first interpretable baseline, the shared metric, and the slices that matter. Then write down where your own judgment agrees or disagrees.

## Extension Exercises

1. Build a linear regression baseline for a small regression dataset and interpret the weight signs.
2. Train a logistic regression classifier and compare threshold choices rather than only the default prediction rule.
3. Run a nearest-neighbor baseline with and without feature scaling and observe the difference.
4. Write a one-page baseline comparison memo for one task you care about.

## Further Reading

- [References](../references.md)
- [Chapter 5. Trees, Ensembles, and Strong Baselines](../chapter-05/README.md)
