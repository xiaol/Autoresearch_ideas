# Method Notes

References:

- Goodfire article: <https://www.goodfire.ai/research/the-world-inside-neural-networks>
- Paper: *Steering Along Manifolds to Control Neural Networks*, arXiv:2605.05115
- Related video reference: <https://youtu.be/dhzSHkCi5tY>

## Original Method

The Goodfire manifold-steering method tests whether a model's internal activation geometry lines up with output behavior geometry.

For a cyclic concept such as weekdays:

1. Collect hidden activations for prompts associated with each concept label.
2. Compute concept centroids in activation space.
3. Reduce the activation dimension with PCA.
4. Fit a smooth periodic curve through the centroids.
5. Patch hidden states along either:
   - the fitted manifold path, or
   - a straight-line baseline.
6. Decode the patched state and measure the output probability distribution over the concept labels.
7. Compare movement in activation space with movement in behavior space.

In this repo, behavior space means the final output probability vector over the target concept labels, not the whole layer activation.

## RWKV Adaptation

Transformers commonly expose a residual stream at each block. RWKV is state based, so there is no exact one-to-one residual-stream object with the same semantics.

The intervention used here patches the last-token block output after the selected RWKV block's time-mix and channel-mix residual updates. This is the closest analogue used for the transformer residual block output in the original method.

## Matched Endpoint Correction

The original draft reproduction allowed linear and manifold paths to have different decoded endpoints in some visualizations. That can make the behavior-space comparison ambiguous: the endpoint difference could come from the target state, not from the path geometry.

The current report uses:

```text
--linear-endpoint-mode matched
```

This makes both methods share identical start and end hidden states. The only difference is the intermediate path.

## Distance and Isometry Metrics

The report includes:

- `r manifold`: correlation between activation-manifold path distance and behavior-manifold path distance.
- `r linear`: the same comparison for straight Euclidean activation distances.
- `dist manifold`: mean behavior-space distance from the manifold-steered path to the fitted behavior manifold.
- `dist linear`: the same distance for the straight-line baseline.

The behavior distance uses probability geometry derived from the output distribution over concept labels. In the implementation, probability vectors are square-root transformed/normalized for Bhattacharyya-style geometry.

## Dimensionality Reduction

The method used to reduce hidden activation dimension is PCA:

```python
from sklearn.decomposition import PCA
coords = PCA(n_components=dim, random_state=0).fit_transform(layer_hidden)
```

For the reported runs:

- `--pca-dim` default is `16` for fitting/intervention.
- 3D visualizations use PCA with `n_components=3`.

This is a linear dimensionality reduction step. No UMAP, t-SNE, autoencoder, or learned nonlinear dimension reducer is used.
