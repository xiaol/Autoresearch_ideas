# MLP Efficiency Visualizer Narration

Source video: `/home/xiaol/2026-06-03 23-19-10.mp4`

Duration: 95.18 seconds

## Term Ledger

| Term | Plain-language definition | Mechanism / input-output | Why it matters here | First timestamp | Visual referent |
| --- | --- | --- | --- | --- | --- |
| MLP / FFN | The feed-forward sublayer inside a transformer block. | Hidden channels are projected up, gated, and projected back down. | This is the part reproduced with the original stacked-neuron visual style. | 00:00 | Four stacked columns in the center graph. |
| Hidden channel | One coordinate of the token's residual-state vector. | A token enters a layer as a vector of hidden values. | The dots are sampled hidden or intermediate channels, not full model weights. | 00:06 | White and gray dots in each column. |
| Gate-up projection | The two expansion projections in a gated feed-forward network. | The hidden vector expands into a larger intermediate vector, then the gate selects useful directions. | It explains why the middle columns are wider and more connected. | 00:14 | Qwen panel: "Gate-Up" and intermediate column. |
| Activation | The current strength of a sampled neuron/channel. | Brighter dots indicate stronger sampled activity for the selected token and step. | It makes the static MLP shape data-dependent without changing the graph layout. | 00:19 | Dot brightness and right-side sample bars. |
| MFLOPs/token | Millions of floating-point operations for one token through the selected mixing layer. | Estimated as `6 * hidden_size * intermediate_size` for dense gated FFN-style mixing. | It gives the compute cost side of efficiency. | 00:24 | Efficiency card on the right panel. |
| Delta/MFLOP | Hidden-state movement divided by estimated dense compute. | Layer output change is normalized by MFLOPs/token. | It is the rough useful-work-per-cost score. | 00:28 | "Delta/MFLOP" field. |
| Active neurons | Estimated count of intermediate channels that carry meaningful activity. | Derived from normalized MLP activity unless true intermediate activations are exported. | It shows concentration: many channels available, fewer channels visibly active. | 00:31 | "Active neurons" field. |
| RWKV | A recurrent language-model architecture that mixes information through state over time. | Replaces attention-style context routing with recurrent time and channel mixing. | It lets the same visualization compare transformer FFN and recurrent mix blocks. | 00:38 | RWKV selector and "RWKV mix" panel. |
| Time mix | RWKV's mechanism for combining current token information with recurrent state. | State-in channels pass through a time-mixing stage before channel mixing. | It is the RWKV analogue to the route through the layer. | 00:43 | "Time Mix" column and right panel. |
| Same prompt comparison | Both models are driven by the same prompt and step controls. | Qwen and RWKV traces advance together while their selected block metrics are displayed side by side. | It makes architectural differences visible under the same input. | 01:05 | Two MLP/RWKV graphs side by side. |

## Narration Beat Map

| Time | Duration | Spoken lines | Visual action | Key terms | Why the visual change matters |
| --- | ---: | --- | --- | --- | --- |
| 00:00-00:10 | 10s | We start with the original MLP visualizer style, but the object is now one language-model block. Each dot is a sampled hidden channel for the selected token. | Qwen MLP-only view, stacked dots, prompt tokens along bottom. | MLP, hidden channel | Establishes that the graph is not a generic network decoration; it is the FFN sublayer for a selected token. |
| 00:10-00:21 | 11s | Read the columns left to right: residual input, gate-up projection, intermediate activation, and residual output. Green and red lines show the strongest sampled routes between stages. | Camera orbits the Qwen graph while token step advances. | Gate-up projection, activation | Explains the visual grammar before metrics appear. |
| 00:21-00:34 | 13s | The efficiency card adds compute. MFLOPs per token estimates dense feed-forward cost. Delta per MFLOP asks how much the layer moved the hidden state for that cost. | Right panel shows MLP Efficiency, active neurons, top-32 coverage, delta/MFLOP. | MFLOPs/token, Delta/MFLOP | Connects the visual activity to a measurement, not only an animation. |
| 00:34-00:47 | 13s | Switching to RWKV keeps the same visual language, but changes the mechanism. The block is now recurrent state, time mix, channel mix, and state output. | Model selector changes to RWKV-7, graph keeps stacked-dot layout. | RWKV, time mix | Shows the adaptation: same visual language, different architecture. |
| 00:47-01:02 | 15s | This RWKV sample has fewer layers and a smaller hidden width, so its per-token compute estimate is lower. The comparison is per-layer, not a full benchmark. | RWKV panel highlights 12 layers, 768 hidden, 14.2 MFLOPs/token. | MFLOPs/token, active neurons | Prevents overclaiming while explaining what can be compared. |
| 01:02-01:15 | 13s | In comparison view, Qwen and RWKV use the same prompt and step control. That makes structure, activity concentration, and cost visible under one input. | Same Prompt Comparison appears with two graphs. | Same prompt comparison | Explains why comparison view is meaningful. |
| 01:15-01:28 | 13s | The right panel lines up the shared metrics: route activity, MLP or channel-mix activity, MFLOPs per token, Delta per MFLOP, active units, and top-32 coverage. | Camera rotates comparison view; right cards show both metric rows. | Delta/MFLOP, active neurons, top-32 coverage | Names the measurable axes visible in the side panel. |
| 01:28-01:35 | 7s | In short: the original stacked-dot MLP view now runs from LLM traces and shows per-block efficiency. | Final comparison view holds. | MLP, LLM trace, efficiency | Synthesizes the work as an LLM-specific extension of the original demo. |

## Voiceover Script

We start with the original MLP visualizer style, but the object is now one language-model block. Each dot is a sampled hidden channel for the selected token.

Read the columns left to right: residual input, gate-up projection, intermediate activation, and residual output. Green and red lines show the strongest sampled routes between stages.

The efficiency card adds compute. MFLOPs per token estimates dense feed-forward cost. Delta per MFLOP asks how much the layer moved the hidden state for that cost.

Switching to RWKV keeps the same visual language, but changes the mechanism. The block is now recurrent state, time mix, channel mix, and state output.

This RWKV sample has fewer layers and a smaller hidden width, so its per-token compute estimate is lower. The comparison is per-layer, not a full benchmark.

In comparison view, Qwen and RWKV use the same prompt and step control. That makes structure, activity concentration, and cost visible under one input.

The right panel lines up the shared metrics: route activity, MLP or channel-mix activity, MFLOPs per token, Delta per MFLOP, active units, and top-32 coverage.

In short: the original stacked-dot MLP view now runs from LLM traces and shows per-block efficiency.
