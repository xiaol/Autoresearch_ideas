# Chapter 3. Data, Labels, and Problem Framing

Many machine learning projects fail before the first model trains.

They fail in meetings where people use the same words to mean different things.

They fail in datasets that answer the wrong question cleanly.

They fail in labels that arrive too late, or too noisily, or only for the cases humans happened to inspect.

They fail because a team says, "We need a churn model," when what they really need is a system for deciding where intervention is worth spending money.

They fail because a team says, "We need fraud detection," when what they actually have is a queue of suspicious transactions, a delayed investigation process, and inconsistent human labels.

This chapter is about that earlier layer of work.

If Chapter 2 made the math legible, Chapter 3 should make the real problem legible.

## 3.1 Case One: The Churn Model That Solved the Wrong Problem

Imagine a subscription software company.

Leadership asks for a churn prediction model. The request sounds straightforward:

- predict which customers will cancel

The data team gathers:

- account age
- usage frequency
- support tickets
- team size
- plan type
- login activity

The ML team is ready to build.

But pause.

What does the business really need?

Often the real question is not:

- who is likely to leave

It is something more operational:

- which customers are both at risk and worth intervening on
- which intervention has a chance of working
- when should the intervention happen
- how do we avoid spending retention effort on customers who would have stayed anyway

Those are not the same problem.

A churn model may accurately predict cancellation risk and still be operationally weak if:

- it flags customers after the point where intervention matters
- it focuses attention on low-value accounts
- it cannot distinguish "at risk" from "already gone in practice"
- it predicts behavior without helping action

The model may be technically correct and strategically unhelpful.

That is why problem framing is not a soft preliminary exercise. It is part of the engineering.

## 3.2 Case Two: Fraud Labels That Arrive Weeks Late

Now consider payments fraud.

The team says:

- classify whether a transaction is fraudulent

Again, that sounds like a standard supervised learning task. But in reality the label pipeline is messy.

A transaction may be:

- obviously legitimate
- automatically blocked
- manually reviewed later
- disputed by a user weeks afterward
- never conclusively resolved

So what exactly is the label?

It might be:

- chargeback received
- investigator marked as fraud
- rule engine escalation
- customer complaint

Each of those definitions changes the training problem.

This is where many beginners, and frankly many teams, make a major mistake:

they act as if the label is the ground truth, when the label is often a delayed, biased, partial trace of a human process.

This matters for at least four reasons:

1. Positive labels may arrive late.
2. Only suspicious cases may be investigated, creating selection bias.
3. Human reviewers may disagree.
4. The cost of a false positive and false negative can be very different.

A model trained on these labels is not simply learning fraud. It may be learning the habits and blind spots of the investigation process.

That is why label thinking is one of the most important habits in machine learning.

## 3.3 What Question Are We Actually Answering?

Before building anything, ask the most uncomfortable question in the room:

**What question is this system actually answering?**

Not what the slide deck says.

Not what the project name implies.

What question does the data, label, and decision pipeline truly define?

Here are examples.

### Stated question

- Which users will churn?

### Actual learnable question

- Which users with this observation window later canceled under our current logging and account lifecycle definitions?

### Stated question

- Which transactions are fraud?

### Actual learnable question

- Which transactions end up receiving one of our available fraud outcomes under the current investigation process?

These reframings may sound less elegant, but they are much more useful. They expose the hidden assumptions.

## 3.4 Prediction Target Versus Business Target

One of the most important distinctions in applied ML is the gap between the **prediction target** and the **business target**.

The prediction target is what the model outputs.

The business target is what the organization actually cares about.

They can be aligned. They are often only partially aligned.

### Churn example

Prediction target:

- probability of cancellation in the next 30 days

Business target:

- increase retained revenue efficiently

A model can be strong on the first and weak on the second.

### Fraud example

Prediction target:

- probability of fraud

Business target:

- reduce fraud loss without crushing legitimate transactions or overwhelming investigators

Again, these are related but not identical.

This gap is where many ML disappointments live. A team builds a high-performing model, only to discover that the optimization target was an imperfect stand-in for the real objective.

The fix is not cynicism. The fix is explicit framing.

Ask:

- what decision will the model influence
- who will use the output
- what action follows the prediction
- what costs or harms matter
- how quickly must the decision happen

If you cannot answer those questions, you are not ready to define the target.

## 3.5 Datasets Are Manufactured Objects

A dangerous beginner belief is that datasets are found.

In reality, datasets are made.

Someone decides:

- what counts as one example
- what time window matters
- which features are available
- what constitutes the label
- which examples are excluded
- how train, validation, and test are split

All of these decisions shape the task.

This is why two teams can work on "the same problem" and still train fundamentally different models. They are not actually using the same task definition.

### Questions to ask about a dataset

- What event creates a row?
- At what timestamp is prediction allowed?
- Which features are known at that moment?
- What future information must be excluded?
- Who is missing from the data?
- Which labels are delayed, noisy, or proxy-based?

These questions are not bureaucratic overhead. They are part of the model.

## 3.6 Proxy Labels and Hidden Assumptions

Many ML systems cannot directly observe what they truly care about.

So they use proxies.

Examples:

- customer complaint as a proxy for bad experience
- chargeback as a proxy for fraud
- watch time as a proxy for content quality
- support ticket resolution as a proxy for user success

Proxy labels are often necessary.

They are also dangerous.

Why?

Because a proxy can be:

- incomplete
- delayed
- biased toward what is easy to measure
- distorted by existing business processes

If you optimize the proxy too aggressively, you may damage the real goal.

This is one of the deepest lessons in applied ML.

A model does not merely learn patterns in the world. It learns patterns in the world as filtered through your measurement system.

That is a much more fragile thing.

## 3.7 Leakage: The Shortcut That Lies

One of the most common and destructive failures in early ML work is **data leakage**.

Leakage happens when information that would not truly be available at prediction time sneaks into training.

Examples:

- a fraud feature that includes the result of later human review
- a churn feature built from behavior that happened after the intervention window
- a support classification feature that includes the final resolution code

Leakage makes a model look smart by teaching it to peek into the future.

That is why leakage feels so good at first:

- the metrics improve
- confidence rises
- everyone feels clever

Then deployment happens, the shortcut disappears, and performance collapses.

The strongest teams develop almost paranoid habits around availability timing and feature provenance. That paranoia is healthy.

## 3.8 Label Quality Is a Systems Problem

People sometimes talk about labels as if they were a static annotation artifact.

In real systems, label quality depends on:

- reviewer guidelines
- reviewer incentives
- reviewer disagreement
- backlog delays
- logging quality
- business policy changes

In other words, labels are produced by a system.

This is especially obvious in:

- fraud review
- moderation
- support routing
- medical or legal adjudication

If the label system changes, the meaning of the label may change too.

That has major consequences for:

- model retraining
- evaluation consistency
- drift monitoring
- fairness analysis

A mature ML team therefore asks not only, "How many labels do we have?" but:

- how were they produced
- by whom
- under what rubric
- with what disagreement rate
- with what delay

Those questions should become instinctive.

## 3.9 A Framing Workflow That Actually Helps

Let us make this operational.

Before starting a project, run a framing workflow with five questions.

### 1. What decision are we supporting?

Not:

- what model do we want to train

But:

- what human or system decision changes because of this output

### 2. What is the prediction moment?

Exactly when is the system allowed to know things?

This question prevents a large fraction of leakage.

### 3. What is the target, and how close is it to the real objective?

Be explicit about whether the label is:

- direct
- delayed
- proxy-based
- human-judged
- incomplete

### 4. What are the main costs of being wrong?

False positives and false negatives rarely cost the same.

The asymmetry should shape both evaluation and deployment design.

### 5. What slice failures would be unacceptable?

Examples:

- one region
- one customer segment
- one payment type
- one product tier

If you cannot name likely failure slices early, you will be less prepared to inspect them later.

## 3.10 Harness Lab: Build a Problem Framing Harness

Now we make the chapter reusable.

The goal is not to have one good conversation once. The goal is to create a harness that improves future project starts.

Here is a simple **Problem Framing Harness**.

### Purpose

Convert a vague ML request into:

- a decision-focused task definition
- a candidate target
- a data and label audit
- a list of known assumptions and risks

### Inputs

- project request
- available data sources
- stakeholder goal
- deployment context

### Required outputs

1. Stated business goal
2. Actual decision to be supported
3. Prediction moment
4. Candidate label and its limitations
5. Feature availability constraints
6. Leakage risks
7. Cost asymmetries
8. Likely failure slices
9. Recommended first baseline

### Minimal workflow

1. Rewrite the request as a decision problem.
2. Define the prediction timestamp.
3. Separate prediction target from business target.
4. Audit label quality and delay.
5. List obvious proxies and hidden assumptions.
6. Identify likely leakage paths.
7. Propose the simplest baseline worth trying.

### Evidence artifact

Before any model training begins, produce a one-page framing memo.

It should include:

- the real decision
- the label definition
- what is unknown
- what could go wrong
- what baseline will be attempted first

That memo is one of the highest-leverage artifacts in an ML project because it prevents technical momentum from racing ahead of conceptual clarity.

## 3.11 A Data Audit Template

In practice, I recommend every early ML project answer a few plain questions.

### Data Audit Questions

1. What creates a row?
2. What timestamp defines the observation window?
3. What information is known at prediction time?
4. Which features might contain future information?
5. How is the label produced?
6. How late can the label arrive?
7. Who is excluded or underrepresented?
8. Which obvious baselines should beat guesswork?

If a team cannot answer these questions, it usually is not ready for model complexity.

## 3.12 Common Failure Modes

### Failure Mode 1. Modeling the Project Name

The team trains a model for the label implied by the ticket title rather than the decision the organization needs.

Fix:

- rewrite every project as a decision-support problem before modeling

### Failure Mode 2. Proxy Blindness

The team forgets that the label is only a partial shadow of the real objective.

Fix:

- name the proxy explicitly and write down what it misses

### Failure Mode 3. Delayed Label Confusion

The team treats late-arriving labels as if they were immediate truth.

Fix:

- document label latency and how it affects training and evaluation

### Failure Mode 4. Leakage by Convenience

Future information sneaks in because it is already sitting in the warehouse.

Fix:

- define the prediction moment first, then audit every feature against it

### Failure Mode 5. Metric Worship Before Framing

The team debates ROC curves or model families before agreeing on the actual task.

Fix:

- force framing, label, and cost discussions to happen before serious model comparison

## Chapter Summary

Many machine learning failures begin before training, because the real work starts with defining the decision, the prediction moment, the label, and the data constraints correctly. Churn and fraud both show how easily a team can optimize the wrong target or trust labels that are delayed, noisy, or proxy-based. Datasets are constructed objects, labels are produced by systems, and leakage is the shortcut that makes a model look smart by cheating. A strong practitioner therefore learns to build a problem framing harness before building a model.

## Use This Skill Now

After this chapter, open [Reader Skill: ML Problem Framer](../reader-skills/ml-problem-framer.md).

Take one vague ML request from your own work or from a product page and turn it into a framing memo. If the memo still feels fuzzy, that is useful evidence that the project is not ready for serious modeling yet.

## Extension Exercises

1. Take an ML problem statement from a product or research article and rewrite it as a decision-support problem.
2. For one case, define the prediction moment and list which features would be illegal because they arrive too late.
3. Choose one proxy label and write two ways it could mislead a model.
4. Draft a one-page framing memo for either churn prediction or fraud detection.

## Further Reading

- [References](../references.md)
- [Chapter 4. First Models: Linear Models and Nearest Neighbors](../chapter-04/README.md)
