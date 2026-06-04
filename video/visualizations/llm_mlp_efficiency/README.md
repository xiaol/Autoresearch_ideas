# LLM Network Visualisation

This module is the AutoResearch visualization package for the "Inside the LLM:
Where Are Facts Stored?" demo direction. It contains the reproducible browser
visualizer, trace exporter, video recorder, narration timing, and 4K conclusion
renderer. Generated MP4/audio outputs are ignored by git; keep final renders
under local `narration-assets/` or the repo-level `video/renders/` area.

This is a local reproduction of the interaction pattern from
`DFin/Neural-Network-Visualisation`, adapted from a small MNIST MLP to compact
LLM telemetry. The browser does not load model weights. It renders a sampled
trace: tokens, generation steps, layer activity, residual flow, and top-token
candidates.

Use **Model trace** to switch between the bundled Qwen3.5-0.8B sample and the
RWKV-7 Goose 0.1B sample. The default view is **MLP Only** for transformer
traces and **RWKV Mix** for RWKV traces. It zooms into the selected block's
feed-forward or recurrent mixing sublayer:

```text
residual hidden channels -> gate/up projection -> SiLU-gated intermediate -> down projection
RWKV state channels -> time mix -> channel mix -> output state channels
```

This keeps the original MLP/FFN visualization method: fixed neuron columns,
activation-driven grayscale node brightness, and strongest sampled green/red
connections between adjacent columns. It samples channels because drawing all
`1024 -> 3584 -> 1024` units and dense links would be unreadable. Use the
**Block Flow** segmented control when you want the broader block overview.
Use **Compare** to draw Qwen3.5-0.8B and RWKV-7 Goose 0.1B side by side from
the exact same prompt text. The comparison view keeps the same stacked-dot
method but uses a compact channel sample for each model so both graphs fit in
one canvas.

The RWKV sample is architecture-only synthetic telemetry. It uses the public
Hugging Face config for `RWKV/RWKV7-Goose-World2.8-0.1B-HF`: 12 layers, hidden
size 768, intermediate size 3072, context length 2048, vocabulary size 65,536,
and `RWKV7ForCausalLM`. The model card describes it as RWKV-7 under
flash-linear-attention format with 191M parameters.

## Input and Output Controls

- **Input token** chooses which token's hidden-state vector feeds the selected
  block FFN. Clicking a token chip does the same thing.
- **Channel sample** changes which hidden/intermediate channels are represented:
  high-activity, even spread, or seeded sample.
- **Output readout** switches the right panel between sampled residual output
  channels and next-token candidates. The FFN itself outputs hidden channels;
  token probabilities are a later readout after the rest of the model and LM
  head.
- In **Compare**, **Build Local Trace** rebuilds both model traces from the same
  prompt and the step controls advance both sides together.

The 3D scene also includes a small input embedding manifold below the FFN. In
the bundled sample trace this is a deterministic projection from token text/id,
used to show the interaction shape. For a real Qwen or RWKV trace, replace it with
PCA/UMAP/t-SNE coordinates exported from actual token embedding or hidden-state
vectors.

## MLP Efficiency Metrics

The selected MLP/RWKV detail panel includes a compact efficiency readout:

- **MFLOPs/token** estimates dense feed-forward cost as
  `6 * hidden_size * intermediate_size` FLOPs per token. This counts multiply
  and add for gate, up, and down projections in a gated FFN.
- **Delta/MFLOP** divides the selected block's hidden-state delta by the dense
  MFLOP estimate, giving a rough "state movement per compute" score.
- **Active neurons** and **Top-32 coverage** describe how concentrated the
  intermediate activity is. Bundled synthetic traces estimate these values from
  normalized MLP activity. The real trace exporter also marks them as estimated
  unless intermediate FFN/channel-mix activations are captured directly.

## Run

```bash
npm install
npm run dev
```

Open the Vite URL in a browser. The default trace uses Qwen3.5-0.8B metadata and
synthetic layer activity so the visualizer works without downloading weights.
The RWKV selector loads `public/traces/rwkv7-01b-sample.json` the same way.

## Record Videos

Start the dev server, then run:

```bash
npm run record:videos
```

This writes three MP4s to `video-captures/`:

```text
qwen35-08b.mp4
rwkv7-01b.mp4
comparison.mp4
```

Use the same custom prompt for all three shots:

```bash
npm run record:videos -- \
  --prompt "Explain how compact language models route information." \
  --duration 10 \
  --fps 24
```

Record only one shot:

```bash
npm run record:videos -- --scenario comparison
```

For a 4K export:

```bash
npm run record:videos:4k
```

This writes 3840x2160 MP4s to `video-captures-4k/`. You can still override the
defaults:

```bash
npm run record:videos:4k -- --scenario comparison --fps 24 --duration 10
```

The recorder orbits the camera by default:

```bash
npm run record:videos:4k -- --camera orbit
```

Use a static camera if you want the original fixed angle:

```bash
npm run record:videos:4k -- --camera static
```

Token changes can look jumpy if the recording sweeps every input token. The
default `--token-mode step` only changes token when the generation step changes.
Use `--token-mode hold` for one fixed token or `--token-mode sweep` when you
intentionally want to scan across all tokens.

## Export a Real Trace

Install the optional Python dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r scripts/requirements.txt
```

Then export a compact Hugging Face trace:

```bash
python scripts/export_llm_trace.py \
  --model Qwen/Qwen3.5-0.8B \
  --prompt "Explain residual streams in one paragraph." \
  --out public/traces/qwen35-08b-real.json \
  --max-new-tokens 6 \
  --trust-remote-code
```

Load the generated JSON from the app using **Load JSON**. If your installed
Transformers build does not yet recognize `qwen3_5`, install a newer
Transformers release or use another supported causal LM with the same script.
For RWKV-7 HF checkpoints, use `trust_remote_code=True`; if the internal
time-mix/channel-mix tensors are not exposed by the installed model class,
export hidden states first and let the visualizer synthesize missing per-block
mix metrics deterministically.

Example RWKV export:

```bash
python scripts/export_llm_trace.py \
  --model RWKV/RWKV7-Goose-World2.8-0.1B-HF \
  --prompt "Explain recurrent state in RWKV." \
  --out public/traces/rwkv7-01b-real.json \
  --max-new-tokens 6 \
  --trust-remote-code
```

## Trace Shape

```json
{
  "schemaVersion": "llm_trace_v1",
  "model": { "name": "Qwen/Qwen3.5-0.8B" },
  "architecture": { "layers": [{ "index": 0, "kind": "linear_attention" }] },
  "tokens": [{ "index": 0, "text": "Hello", "source": "prompt" }],
  "steps": [
    {
      "activeTokenIndex": 0,
      "topTokens": [{ "token": " world", "probability": 0.42 }],
      "layers": [
        {
          "residualNorm": 0.7,
          "attention": 0.3,
          "mlp": 0.5,
          "efficiency": {
            "megaFlopsPerToken": 22.02,
            "activeFraction": 0.23,
            "activeNeurons": 824,
            "topK": 32,
            "topKCoverage": 0.64,
            "deltaPerMFlop": 0.018,
            "estimated": true
          }
        }
      ]
    }
  ]
}
```

The visualizer accepts missing layer metrics and fills them deterministically,
which is useful for architecture-only demos.
