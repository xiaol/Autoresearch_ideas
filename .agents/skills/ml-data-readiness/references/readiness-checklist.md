# Data Readiness Checklist

## Core Questions

1. What creates a row?
2. What timestamp defines prediction time?
3. Which fields are known at that moment?
4. Which joins could duplicate or distort data?
5. Which features may be stale?
6. Which fields may leak future information?
7. Is the serving path equivalent to the training path?
8. Who owns the critical features?

## Suggested Output

```md
Sources:

Critical Fields:

Timestamp Semantics:

Freshness Risks:

Join / Duplication Risks:

Leakage Risks:

Train-Serve Risks:

Lineage / Ownership:

Readiness Verdict:

Required Fixes:
```
