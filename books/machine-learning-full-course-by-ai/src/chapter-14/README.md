# Chapter 14. Training, Serving, and MLOps

A model that performs well in development but fails in deployment has not succeeded.

This sounds obvious, yet many learning paths still treat deployment as an afterthought. In real work, deployment is where the system becomes accountable.

This chapter is about the engineering required to move from a trained artifact to a dependable service or workflow.

## 14.1 Case One: Launching a Support Assistant

Imagine an enterprise support assistant going live.

It can:

- summarize tickets
- retrieve policies
- draft answers

The risks are immediate:

- latency frustrates agents
- bad answers create customer harm
- hallucinated policy advice creates trust problems
- cost per request rises unexpectedly

The technical question is no longer only:

- does the model work

It is:

- can the whole serving system operate safely, cheaply, and observably

## 14.2 Case Two: Drift After City Expansion

Now return to our delivery-time prediction case.

The model was trained on one city.

The company expands into a new city with different:

- road layouts
- traffic patterns
- restaurant density
- rider availability

Suddenly performance degrades.

This is not necessarily a training bug. It may be deployment reality exposing distribution shift.

That is why serving and monitoring must be part of the original system design.

## 14.3 Batch Versus Online Inference

One of the first deployment choices is:

- batch or online inference

### Batch inference

Good when:

- latency is not critical
- predictions can be prepared ahead of time
- cost efficiency matters more than immediacy

### Online inference

Good when:

- decisions must react immediately
- user interactions require low latency
- freshness is critical

This choice affects infrastructure, monitoring, cost, and failure modes.

## 14.4 Model Packaging and Registries

A trained model is not only weights. To deploy responsibly, you need:

- model artifact
- configuration
- feature assumptions
- version identity
- evaluation record

Registries and artifact tracking matter because teams need to know:

- which model is live
- what data it came from
- what metrics justified it
- how to roll back if needed

Without this, deployment turns into folklore.

## 14.5 Rollout Strategy and Deployment Gates

Strong teams rarely move from offline success to full rollout in one leap.

They use gates such as:

- offline evaluation pass
- shadow mode
- limited traffic rollout
- monitored ramp-up

These gates exist because the real system is always richer and messier than the dev setup.

Deployment gates are not hesitation. They are structured humility.

## 14.6 Latency, Cost, and Observability

A model can be excellent and still unacceptable if:

- it is too slow
- it is too expensive
- no one can observe its behavior

This is especially true for foundation-model-based services.

So serving design must ask:

- what is the latency budget
- what is the cost per request or prediction
- what signals will reveal trouble

Observability means you can see:

- traffic
- errors
- confidence or quality shifts
- slice degradation
- resource consumption

Without observability, production ML becomes guesswork under pressure.

## 14.7 Monitoring for Drift and Degradation

Monitoring is not only uptime.

For ML, it should also include:

- input distribution changes
- feature freshness problems
- prediction distribution shifts
- slice-level quality degradation
- downstream business outcome changes

This is where Chapter 6 and Chapter 13 come back together.

Good monitoring connects:

- evaluation logic
- data reliability
- production behavior

## 14.8 Rollback and Incident Response

Every serious ML system should have an answer to:

- what happens when the model goes wrong

That answer might include:

- switch to previous model
- fall back to rule-based logic
- disable one capability
- escalate to human review

Rollback is not a sign of failure. It is proof that the team expected reality.

Incident response matters because deployment is not only a launch event. It is an ongoing contract with the environment.

## 14.9 Harness Lab: Build a Launch Readiness Harness

Here is a simple **Launch Readiness Harness**.

### Purpose

Determine whether an ML system is ready for production rollout under real constraints.

### Inputs

- offline evaluation
- serving design
- latency and cost estimates
- monitoring plan
- rollback plan

### Required outputs

1. Serving mode decision
2. Deployment gate checklist
3. Monitoring requirements
4. Cost and latency risk summary
5. Rollback or fallback plan
6. Recommendation for launch, limited rollout, or delay

### Minimal workflow

1. Restate the operational stakes.
2. Decide batch versus online requirements.
3. Check whether observability is sufficient.
4. Review drift, slice, and cost monitoring.
5. Verify rollback or human fallback exists.
6. Choose the rollout strategy.

### Evidence artifact

Produce a launch readiness review that states:

- what is ready
- what is risky
- what will be monitored
- what triggers rollback

This is the kind of artifact that turns deployment from enthusiasm into responsibility.

## 14.10 Common Failure Modes

### Failure Mode 1. Offline Victory Syndrome

The team treats offline metrics as proof of production readiness.

Fix:

- require deployment gates and live monitoring plans

### Failure Mode 2. Observability Neglect

The system ships without the signals needed to detect degradation.

Fix:

- make observability part of launch scope, not a future enhancement

### Failure Mode 3. Cost Blindness

The system works technically but becomes too expensive to operate responsibly.

Fix:

- model cost as part of system quality

### Failure Mode 4. No Rollback Story

The team assumes failure will be rare enough not to plan for.

Fix:

- define rollback and fallback before launch

### Failure Mode 5. Drift as Surprise

The environment changes and the team reacts as if this were an exceptional event.

Fix:

- treat drift as a normal production condition

## Chapter Summary

Production ML is not the moment when a good model gets copied onto a server. It is the disciplined practice of packaging, serving, monitoring, gating, and rolling back systems under real constraints. Latency, cost, observability, and drift are not secondary details. They are part of the model's truth in the world.

## Use This Skill Now

After this chapter, open [Reader Skill: ML Launch Readiness](../reader-skills/ml-launch-readiness.md).

Use it on one system that feels "almost ready." The point is not to block launches forever. The point is to learn which missing gates, monitoring signals, fallback paths, or cost assumptions actually matter before real users are affected.

## Extension Exercises

1. Decide whether a given task should use batch or online inference and justify it.
2. Write down five signals you would monitor for a support assistant.
3. Draft a rollback plan for a delivery prediction system after distribution shift.
4. Create a launch readiness checklist for one ML system.

## Further Reading

- [References](../references.md)
- [Chapter 15. Responsible AI, Safety, and Human Feedback](../chapter-15/README.md)
