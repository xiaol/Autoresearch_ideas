# Higher-Order Benchmark Audit

## What was wrong

The original higher-order multi-task curriculum reused the same sequence format for:
- `binding-copy`
- `binding-affine`
- `binding-gate`
- `binding-lookup`

but did **not** include any task identifier in the input.

That made the benchmark ambiguous:
- the same `(tag, A, B, C)` sequence could legitimately map to four different labels
- a single model trained across tasks had no way to infer which label rule to apply

As a result, the earlier mixed-curriculum results were not a valid measure of multi-task triple-interaction reasoning.

## Correction

The benchmark now prepends an explicit task token immediately after `BOS`.

This makes the label function well-defined for mixed training and evaluation:
- sequence content provides the queried `(tag, A, B, C)` binding
- task token selects which composition rule should be applied

## Strongest result so far

On the corrected staged benchmark in `results/_staged_higher_order_typedlatent_v2`:
- `transformer-triple` reaches `12.5% / 6.1% / 7.4% / 9.5%` on `copy / affine / gate / lookup`
- `typed-latent` reaches `100% / 100% / 100% / 100%`

## Interpretation

The `typed-latent` model is benchmark-aware:
- it compresses the sequence into a typed latent state keyed by `(tag, role)`
- it uses exact symbolic heads for `copy`, `affine`, and `gate`
- it uses a learned compressed lookup table for `binding-lookup`

This is best read as:
- an **internal benchmark SOTA** on the corrected task
- a constructive existence proof that latent-state compression can realize true triple-token interactions

It is **not** evidence of generic language-model SOTA.
