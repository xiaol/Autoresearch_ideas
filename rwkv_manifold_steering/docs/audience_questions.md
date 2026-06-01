# Audience Questions

## What method is used to reduce the embedding dimension?

We use PCA, Principal Component Analysis.

In the experiment code, hidden activations are projected with:

```python
from sklearn.decomposition import PCA
coords = PCA(n_components=dim, random_state=0).fit_transform(layer_hidden)
```

For the reported steering experiments, `dim` is controlled by `--pca-dim`, with default `16`.

For 3D figures, we use PCA again with `n_components=3`, only so the high-dimensional activation/behavior geometry can be plotted.

So the short answer is:

> The embedding dimension is reduced with PCA. We use 16 PCA dimensions for manifold fitting/steering by default, and 3 PCA dimensions for visualization. We did not use UMAP, t-SNE, or a learned autoencoder in these reported runs.

## What do the square and dot mean in the behavior-space GIFs?

- Square: manifold steering path.
- Dot/circle: linear steering baseline.
- Connected labeled points: natural concept outputs, projected into behavior space.

If the square follows the connected concept path while the dot cuts away, it suggests that moving along the activation manifold preserves more natural output behavior than moving along the straight activation chord.

## Is behavior space the whole layer output?

No. In these experiments, behavior space is the final output probability distribution over the cyclic concept labels.

For weekdays, the behavior vector is over:

```text
Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
```

For months, the behavior vector is over:

```text
January, February, ..., December
```

The hidden activation is patched at an internal layer/block, but the behavior point is computed after decoding the model's final logits into concept-label probabilities.

## Why are the tiny-model accuracies low?

The small RWKV/Qwen models are near chance on the raw weekday/month arithmetic prompts. This result should be interpreted as a geometry and intervention sanity check, not as evidence that these small models solve the tasks.

