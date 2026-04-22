# Chapter 13. Data Engineering for ML Teams

Many ML conversations are more glamorous than the work that actually determines whether the system survives.

That glamorous layer includes:

- model architecture
- training strategy
- benchmark performance

The survival layer looks more ordinary:

- schemas
- joins
- timestamps
- freshness
- feature lineage
- reproducibility

This chapter argues that many model failures are really data engineering failures wearing model-shaped clothing.

## 13.1 Case One: The Fraud Feature That Changed Meaning

Imagine a fraud model with a feature called `device_risk_score`.

The model has relied on it for months.

Then an upstream service changes how that score is computed. The field name stays the same. Documentation lags behind. Retraining proceeds normally.

Soon performance degrades, but the issue is not obvious because:

- the pipeline still runs
- the model still trains
- dashboards still move

The model did not suddenly become bad at learning. The meaning of the data changed underneath it.

This is why data engineering is part of model correctness.

## 13.2 Case Two: Broken Joins in Recommendation Training

Now imagine a recommendation system built from:

- user events
- item metadata
- impression logs
- conversion events

One bad join duplicates impressions for a subset of users.

Nothing crashes.

The training dataset simply becomes wrong.

The model may learn nonsense confidently, and the team may spend days tuning hyperparameters instead of questioning the pipeline.

This is a brutal but important lesson:

pipeline bugs often masquerade as modeling problems.

## 13.3 Schemas Are Agreements

At a high level, a schema is not just a technical format. It is an agreement about meaning.

If one team thinks a timestamp means event creation time and another thinks it means ingestion time, the pipeline may still execute while the meaning breaks silently.

This is why schema discipline matters.

You want:

- explicit fields
- explicit types
- explicit units
- explicit timestamp semantics

Without that, ML systems become fragile because they depend on hidden assumptions traveling correctly across teams and services.

## 13.4 Data Ingestion and Transformation

By the time a feature reaches a training job, it has usually passed through many steps:

- ingestion
- cleaning
- transformation
- joining
- aggregation
- filtering

Each step can introduce:

- bugs
- drift
- latency
- mismatch between train and serving logic

This is why data pipelines deserve the same seriousness as model code.

## 13.5 Feature Pipelines and Feature Freshness

Many features are time-sensitive.

Examples:

- user activity counts
- recent merchant behavior
- account health signals
- rolling support-ticket volume

If these features arrive late or are computed inconsistently between training and serving, the model may degrade in ways that look mysterious.

Feature freshness is therefore part of model quality.

So is feature consistency:

- was the feature computed the same way offline and online

This is where feature stores, shared definitions, or clear pipeline ownership can matter a lot.

## 13.6 Validation Before Training

One of the best habits in ML engineering is to validate the data before training anything expensive.

That can include checks for:

- missingness spikes
- impossible values
- join explosions
- class balance shifts
- timestamp anomalies
- schema changes

The point is not perfection. The point is to catch silent failure early, when the fix is still cheap.

## 13.7 Lineage and Versioning

When a model behaves badly, you need to answer:

- what data produced this model
- what pipeline version created that data
- what feature definitions were in force

If you cannot answer those questions, debugging becomes archaeology.

Lineage and versioning turn archaeology into engineering.

They let teams trace:

- where the data came from
- how it was transformed
- which model depended on it

This is one reason mature ML teams treat data assets more like code than like disposable files.

## 13.8 Collaboration Between Data and ML Teams

Machine learning systems often fail socially before they fail technically.

For example:

- data engineers change a field without understanding model dependencies
- ML engineers depend on a feature they do not truly understand
- analytics definitions differ from production definitions

Good collaboration reduces these mismatches.

That means:

- shared ownership on critical features
- explicit contracts
- incident reviews
- visible documentation

This may sound procedural, but it is one of the main reasons some teams scale ML reliably and others do not.

## 13.9 Harness Lab: Build a Data Readiness Harness

Here is a simple **Data Readiness Harness**.

### Purpose

Determine whether a dataset and feature pipeline are trustworthy enough for serious modeling.

### Inputs

- raw data sources
- transformed dataset
- feature definitions
- split strategy
- recent pipeline change history

### Required outputs

1. Schema summary
2. Critical timestamp definitions
3. Known freshness constraints
4. Validation checks passed or failed
5. Feature train-serving consistency risks
6. Ownership and lineage notes

### Minimal workflow

1. List the upstream sources.
2. Record the meaning of each critical field.
3. Check for obvious distribution or missingness anomalies.
4. Audit the timestamp logic for leakage and freshness issues.
5. Record the transformation and ownership path for critical features.

### Evidence artifact

Produce a one-page data readiness review before major training cycles.

It should state:

- what changed
- what is trustworthy
- what is uncertain
- what must be monitored after training

This review turns data quality from background hope into explicit engineering evidence.

## 13.10 Common Failure Modes

### Failure Mode 1. Schema Complacency

The field name stayed the same, so the team assumes the meaning stayed the same.

Fix:

- treat schemas as contracts about meaning, not only shape

### Failure Mode 2. Join Blindness

The pipeline creates duplicates or mismatched entities without anyone noticing.

Fix:

- validate row counts, entity uniqueness, and join assumptions explicitly

### Failure Mode 3. Freshness Neglect

The model depends on features that are stale at serving time.

Fix:

- track freshness as part of feature quality

### Failure Mode 4. Train-Serve Mismatch

Features are computed differently in offline training and live inference.

Fix:

- audit the feature path end to end

### Failure Mode 5. Debugging Without Lineage

The team cannot trace which data and transformations created the current model.

Fix:

- log versions, lineage, and ownership as first-class metadata

## Chapter Summary

Reliable machine learning depends on reliable data engineering. Schemas define meaning, pipelines shape the actual task, joins can quietly corrupt truth, freshness changes model utility, and lineage makes debugging possible. Strong ML teams do not treat data as a passive input. They treat it as an engineered dependency that deserves contracts, validation, and ownership.

## Use This Skill Now

After this chapter, open [Reader Skill: ML Data Readiness](../reader-skills/ml-data-readiness.md).

Take one dataset that looks convenient and force yourself to review timestamps, joins, freshness, and train-serve mismatches before you let yourself model it. This is one of the simplest ways to avoid expensive self-deception.

## Extension Exercises

1. Write down three fields in a hypothetical pipeline whose meaning could silently drift.
2. Describe one join bug that could make a model look better than it really is.
3. Draft a small data readiness review for a fraud or recommendation dataset.
4. Explain how feature freshness could affect a live ML system.

## Further Reading

- [References](../references.md)
- [Chapter 14. Training, Serving, and MLOps](../chapter-14/README.md)
