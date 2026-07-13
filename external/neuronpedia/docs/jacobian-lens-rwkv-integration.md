# Jacobian Lens, RWKV, and the JLens UI

The RWKV adapter implements the same JLens stream contract as the
TransformerLens inference server while using RWKV-native recurrent inference.
It supports both a direct Logit Lens and a fitted Jacobian Lens for RWKV-7 G1
checkpoints.

## Components

- `apps/inference/neuronpedia_inference/adapters/rwkv_jlens_adapter.py` serves
  `/v1/lens/prompt` and `/health`.
- `apps/inference/neuronpedia_inference/adapters/fit_rwkv_jlens.py` fits an RWKV
  Jacobian artifact offline.
- `apps/webapp/app/api/lens/prompt/route.ts` proxies the browser request to the
  registered inference host.
- `apps/webapp/components/jlens/` renders the same prompt, layer readouts,
  steering controls, and chat transcript used by other JLens backends.

The local defaults are an adapter at `http://127.0.0.1:5003` and a web proxy at
`http://localhost:3100/api/lens/prompt`.

## RWKV Jacobian Fitting

RWKV's production recurrent CUDA path is forward-only. The offline fitter
therefore mirrors the full RWKV-7 recurrence with differentiable PyTorch
operations. Before fitting, it compares every captured block output and final
logits against the production CUDA path. `--parity-only` runs this check without
writing an artifact.

```bash
python apps/inference/neuronpedia_inference/adapters/fit_rwkv_jlens.py \
  --parity-only \
  --max-seq-len 8

python apps/inference/neuronpedia_inference/adapters/fit_rwkv_jlens.py \
  --model-path /path/to/model.pth \
  --max-seq-len 48 \
  --dim-batch 4
```

Fitting currently requires CUDA. It uses a streaming same-position Jacobian:
for each token, recurrent state and shift paths inherited from earlier tokens
are detached, and the final-layer output at that position is differentiated
with respect to the source block output at the same position. Those Jacobians
are averaged over calibration positions and prompts. This matches live RWKV
readout, where one matrix transports the current token's hidden row; a
cross-position estimator also mixes in indirect effects propagated through
future recurrent states and performed poorly in held-out validation. The fitter
uses every non-final block by default. Use `--source-layers 0,2,4` to fit a
subset, repeat `--prompt` to supply calibration texts, and use `--output` to
override the destination.

The default artifact is next to the model checkpoint:

```text
<model-stem>_jacobian_lens.pt
```

The adapter resolves the same path automatically. `RWKV_JLENS_PATH` or the
adapter's `--jlens-path` option selects a different artifact.

## Artifact Contract

The fitter writes a PyTorch checkpoint whose transport values are:

```text
J: {source_layer: Tensor[d_model, d_model]}
source_means: {source_layer: Tensor[d_model]}
target_mean: Tensor[d_model]
```

For a source-layer row vector `h`, the artifact stores a source mean `mu_l` and
final-layer mean `mu_final`. The adapter transports it toward the final layer as
`(h - mu_l) @ J[layer].T + mu_final`, then applies RWKV's final layer norm and
output head. Centering is required because a Jacobian maps perturbations rather
than absolute coordinates; applying `J` to RWKV's raw residual origin performs
poorly on held-out text. The affine offset does not affect Jacobian steering,
which continues to use the differential map `J` alone.
The final model layer is an identity transport, so it is always available even
though the artifact must not contain a matrix for it. Jacobian steering uses
the corresponding row-vector direction `unembedding @ J[layer]`.

The loader rejects an artifact unless all of these match the served model:

- `format_version == 2`
- `architecture == "rwkv7-g1"`
- `activation_site == "block_output"`
- `transport == "affine_centered"`
- `target_layer == n_layer - 1`
- `n_layer` matches the checkpoint
- `d_model` matches the checkpoint embedding width
- `tokenizer == "rwkv_vocab_v20230424"`
- `estimator == "same_position_mean"`
- `model_sha256` matches the exact `.pth` file
- `source_layers` exactly matches the keys in `J`
- `source_means` has one finite `[d_model]` vector for every key in `J`
- `target_mean` is one finite `[d_model]` vector
- every matrix has shape `[d_model, d_model]` and only finite values

The artifact also records calibration provenance including `n_prompts`,
`calibration_sha256`, `max_seq_len`, and `skip_first`.

`GET /health` reports `jlens_path`, `jlens_status`, `jlens_n_prompts`, and any
`jlens_error`. Its `supports` list includes `JACOBIAN_LENS` only after a valid
artifact is loaded. Any request that includes an unavailable lens type fails
with the artifact error instead of silently dropping it. A Logit-only request
continues to work without an artifact. The adapter never labels a Logit Lens
result as a Jacobian Lens result.

## Stream Contract

The browser consumes NDJSON from `/api/lens/prompt`, which proxies the
adapter's `/v1/lens/prompt`. A normal RWKV request asks for both lens types:

```json
{
  "modelId": "rwkv7-g1d-0-1b",
  "inferenceEngine": "RWKV",
  "chat": [{ "role": "user", "content": "What is a Jacobian?" }],
  "type": ["JACOBIAN_LENS", "LOGIT_LENS"],
  "topN": 8,
  "temperature": 0,
  "numCompletionTokens": 32,
  "stream": true
}
```

The response keeps the standard message order:

```json
{"kind":"meta","model":"rwkv7-g1d-0-1b","types":["JACOBIAN_LENS","LOGIT_LENS"],"layers_by_type":{"JACOBIAN_LENS":[0,1,11],"LOGIT_LENS":[0,1,11]},"top_n":8,"prompt_len":12,"num_completion_tokens":32,"temperature":0,"prepend_bos":false,"reuse_len":0}
{"kind":"prompt","tokens":[{"position":0,"token":"User","token_bytes":[85,115,101,114],"id":123,"is_generated":false}]}
{"kind":"token","position":0,"token":"User","token_bytes":[85,115,101,114],"id":123,"is_generated":false,"results":[{"type":"JACOBIAN_LENS","top_tokens":[["The"]],"top_probs":[[0.12]]},{"type":"LOGIT_LENS","top_tokens":[["A"]],"top_probs":[[0.1]]}]}
{"kind":"done","seq_len":30,"prompt_len":12,"vocab_size":65536,"completion":"A Jacobian is ..."}
```

Required invariants are unchanged:

- `layers_by_type[type]` aligns one-to-one with that type's `top_tokens`,
  `top_probs`, `top_token_ids`, and `top_ranks` rows.
- The fitted source layers and final layer are returned for
  `JACOBIAN_LENS`; requested Logit Lens layers use the same block-output site.
- `top_probs` use the full-vocabulary softmax denominator. Non-word filtering
  changes candidate selection, not those probabilities.
- Every position carries `position`, `id`, `token`, and `is_generated`.
- `input_token_ids` performs an exact replay without generation.
- `cached_token_ids` enables prefix reuse, reported through `reuse_len`.
- Steering, ablation, and swap produce an ordinary
  `meta`/`prompt`/`token`/`done` stream for the intervened forward pass.

RWKV uses a byte-level tokenizer. Its prompt and token records additionally
carry optional `token_bytes: number[]`. The frontend incrementally decodes those
bytes, so a Unicode code point split across multiple token ids appears once,
on the final contributing token, instead of as several replacement characters.
Other model families and older backends that omit `token_bytes` keep their
existing token strings unchanged.

## G1 Chat Formatting

Register an instruction-tuned RWKV model with `instruct: true`; otherwise the
JLens page intentionally selects completion mode instead of chat mode. The
local registration script does this for `rwkv7-g1d-0-1b`.

The adapter formats structured chat according to the checkpoint-specific
RWKV7-G1x template:

- Turns use plain `System:`, `User:`, and `Assistant:` headers.
- Completed turns are separated by exactly two newlines.
- System and user text normalize line endings and collapse embedded blank-line
  separators; assistant history preserves paragraph breaks.
- A trailing assistant message is treated as a prefill and is not replaced.
- Default generation appends `Assistant: <think>\n</think`.
- Thinking mode appends `Assistant: <think`.
- The missing final `>` in both scaffolds is intentional: the G1 prompt guide
  leaves it for the model to generate.
- The prompt is encoded directly and does not prepend token `0`.

The client groups these plain-text role headers back into user and assistant
bubbles; RWKV does not use the ChatML fallback.

## End-of-Turn Behavior

RWKV token `261` represents the blank-line end-of-turn separator. Generation
stops before token `261` is forwarded or displayed. The adapter also stops
before token `0` (`<|endoftext|>`) and recognizes a blank-line separator split
across two generated pieces. Consequently the assistant response ends before a
new `User:` header and cannot bleed into a model-invented next turn.

## Local Registration and Routing

The webapp supports `InferenceEngine.RWKV` and `InferenceEngine.RWKV_MS`.
`apps/webapp/scripts/register-rwkv-jlens-dev.ts` upserts the local model as an
instruction model and registers its host. Environment variables customize the
model id, host URL, engine, dimensions, and display metadata.

When `USE_LOCALHOST_INFERENCE=true`, RWKV and RWKV-MS requests use
`LOCALHOST_RWKV_INFERENCE_HOST` (`http://127.0.0.1:5003` by default). Direct API
callers should pass `inferenceEngine` when more than one backend can serve the
same model id.

For RWKV-MS online memory, keep mutable memory diagnostics in Neuronpedia
source sets such as `rwkv-ms-read`, `rwkv-ms-route`, and `rwkv-ms-slot` rather
than changing the JLens stream schema.
