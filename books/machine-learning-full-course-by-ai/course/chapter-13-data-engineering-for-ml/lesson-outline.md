# Lesson Outline

- Lesson title: Data Engineering for ML Teams
- Lesson goal: show that many model failures are actually data-system failures
- Anchor case: broken joins and stale features
- Main skill or harness: ML Data Readiness
- Primary visual: source -> transformation -> feature -> train/serve path
- On-screen artifact: data readiness review
- Viewer takeaway: data should be trusted only after timestamp, join, freshness, and lineage checks

## Teaching Flow

1. Open with a feature that silently changed meaning.
2. Explain schema semantics, freshness, joins, and lineage.
3. Show train-serve mismatch risk.
4. Walk through a data readiness review.
5. End with the Data Readiness harness.
