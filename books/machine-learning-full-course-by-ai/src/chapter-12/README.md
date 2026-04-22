# Chapter 12. Recommendation, Ranking, and Decision Systems

So far in the book, most models have predicted things.

Now we shift to systems that do more than predict. They choose, rank, prioritize, and shape what people see.

That shift changes everything.

Once a model participates in action selection, it starts affecting the data it will later receive. It becomes part of a feedback loop.

This chapter is about learning to think inside those loops.

## 12.1 Case: The Feed That Optimizes the Wrong Future

Imagine a short-video or e-commerce feed.

The ranking system decides what users see first.

The simplest objective might be:

- maximize click-through rate

At first this seems sensible.

But after some time, the product team notices:

- diversity falls
- user trust drops
- narrow content styles dominate
- long-term satisfaction becomes harder to measure

What happened?

The model did not merely reflect preference. It shaped it.

This is the core challenge of ranking and decision systems:

they act on the world and thereby change the world they later model.

## 12.2 From Prediction to Action

A recommendation or ranking system often has at least two layers:

- estimate likely usefulness or relevance
- decide what to show and in what order

Those are not the same task.

Prediction asks:

- what is likely

Decision asks:

- what should we do given what is likely

That second question introduces incentives, costs, uncertainty, and delayed consequences.

## 12.3 Candidate Generation and Ranking

Many large recommendation systems work in stages.

### Candidate generation

Produce a smaller set of plausible items from a huge universe.

### Ranking

Order those candidates according to some objective or set of objectives.

This multi-stage design matters because:

- the search space is too large
- latency matters
- different stages may optimize different tradeoffs

Once you understand this pipeline, recommendation systems feel less like mystical personalization engines and more like structured decision systems under resource constraints.

## 12.4 Metrics That Fight Each Other

Recommendation and ranking systems often force teams to face a difficult truth:

- optimizing one metric can damage another

For example:

- clicks may rise while satisfaction falls
- watch time may rise while trust falls
- conversion may rise while catalog diversity collapses

This is not an edge case. It is the normal state of multi-objective systems.

That is why ranking work requires more product judgment than many beginners expect.

## 12.5 Feedback Loops

Feedback loops are where recommendation systems become especially tricky.

If the model shows more of one item type, users will interact more with that item type. Then the system learns from those interactions and becomes even more confident that this content should dominate.

This can create:

- popularity spirals
- exposure bias
- narrow content funnels
- unfair visibility patterns

A model in this setting is not just measuring behavior. It is partially producing it.

That should change how we evaluate it.

## 12.6 Exploration Versus Exploitation

A mature decision system must balance:

- exploitation: show what seems best based on current knowledge
- exploration: try alternatives to learn whether something better exists

If the system only exploits:

- it may become trapped in a narrow local optimum

If it explores too aggressively:

- user experience may degrade

This tradeoff appears in many systems:

- recommenders
- ads
- search ranking
- notification timing
- reinforcement-style product loops

Even if you do not specialize in these systems, learning this tension will improve your engineering judgment.

## 12.7 Offline Versus Online Evaluation

Offline metrics matter, but they are not the whole story.

Offline evaluation can tell you:

- how well the system predicts historical behavior
- whether a ranking model beats a baseline on logged data

But it often cannot fully reveal:

- how user behavior changes under new exposure
- how incentives shift
- whether long-term trust rises or falls

This is why online evaluation, A/B testing, and careful rollout become so important in ranking systems.

Again, the main lesson is not "offline is bad." It is:

- offline results are an incomplete window into systems that change user behavior

## 12.8 Harness Lab: Build a Ranking System Review Harness

Here is a simple **Ranking System Review Harness**.

### Purpose

Evaluate a recommendation or ranking system as a decision loop rather than only as a predictive model.

### Inputs

- ranking objective
- candidate generation design
- offline metrics
- online metrics or rollout data
- known product risks

### Required outputs

1. Primary objective and its limitations
2. Secondary metrics that guard against damage
3. Main feedback loop risks
4. Whether exploration is sufficient
5. Whether offline gains are likely to hold online
6. Recommendation for rollout, redesign, or further testing

### Minimal workflow

1. State what the system is optimizing.
2. Ask what that objective leaves out.
3. Identify who could be overexposed or underexposed.
4. Review offline and online evidence separately.
5. Decide whether the system is learning healthy or distorted behavior.

### Evidence artifact

Write a short ranking review memo with:

- objective
- metric stack
- feedback loop risks
- online uncertainties
- rollout recommendation

This memo pushes the team to think like system designers, not only metric collectors.

## 12.9 Common Failure Modes

### Failure Mode 1. Click Worship

The team optimizes one engagement metric as if it were the whole product.

Fix:

- use a metric stack that reflects broader product outcomes

### Failure Mode 2. Offline Delusion

The team assumes offline ranking wins will automatically improve the live system.

Fix:

- separate predictive evaluation from behavioral evaluation

### Failure Mode 3. Feedback Loop Blindness

The team forgets the model changes the data it later sees.

Fix:

- treat ranking systems as intervention systems, not passive observers

### Failure Mode 4. No Exploration Budget

The system overcommits to what it already knows and stops learning.

Fix:

- build explicit exploration logic and monitoring

### Failure Mode 5. Candidate Generation Neglect

The team debates ranking sophistication while the candidate pool is already biased or weak.

Fix:

- evaluate the full pipeline, not only the final scorer

## Chapter Summary

Recommendation, ranking, and decision systems are different from ordinary predictive models because they act on the world and change the feedback they later receive. That introduces incentives, feedback loops, exposure effects, and tension between offline and online evaluation. Strong engineers therefore evaluate these systems not only by predictive accuracy, but by how they shape behavior over time.

## Use This Skill Now

After this chapter, open [Reader Skill: ML Ranking Review](../reader-skills/ml-ranking-review.md).

Use it on one feed, search, or recommendation loop. The main goal is to practice thinking beyond the model score and toward exploration, objectives, feedback loops, and rollout consequences.

## Extension Exercises

1. List three ways a ranking system can improve one metric while harming the product.
2. Explain why offline evaluation is necessary but insufficient for recommenders.
3. Write a short ranking review memo for a feed or search system you know.
4. Describe one exploration strategy and one risk it introduces.

## Further Reading

- [References](../references.md)
- [Chapter 13. Data Engineering for ML Teams](../chapter-13/README.md)
