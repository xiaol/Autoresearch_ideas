# Results Summary

These are the matched-endpoint runs saved in `reports/manifold_report/metrics.json`.

| task | model | layer | base acc | r manifold | r linear | dist manifold | dist linear |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weekday | RWKV-7 0.1B | 11 | 0.143 | 0.949 | 0.891 | 0.0128 | 0.0133 |
| weekday | Qwen3.5 0.8B | 2 | 0.143 | 0.857 | 0.537 | 0.5203 | 0.5228 |
| month | RWKV-7 0.1B | 5 | 0.083 | 0.989 | 0.210 | 1.9228 | 1.9216 |
| month | Qwen3.5 0.8B | 20 | 0.083 | 0.915 | 0.744 | 0.0638 | 0.0639 |

## Interpretation

The most useful signal is not raw task accuracy. These models are too small for strong weekday/month arithmetic performance.

The useful signal is geometric:

- For weekdays, both RWKV and Qwen show stronger activation-to-behavior correlation under manifold distance than under linear distance.
- For months, RWKV shows a large gap between manifold distance correlation and linear distance correlation.
- All corrected runs have zero endpoint deltas between linear and manifold paths, so path differences are not endpoint artifacts.

## Caveat

This is a reproduction and extension sanity check. A stronger claim would need larger models, stronger prompts/tasks, and repeated seeds/model revisions.

