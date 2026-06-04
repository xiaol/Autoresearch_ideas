# MLP Efficiency Conclusion Insert

This insert is appended after the existing narrated visualizer recording.

## Term Ledger

| Term | Plain-language definition | Mechanism / input-output | Why it matters | First timestamp in insert | Visual referent |
| --- | --- | --- | --- | --- | --- |
| Per-layer diagnostic | A measurement for one selected model block, token, and step. | It compares hidden-state movement against estimated compute inside the selected block. | It prevents overclaiming from one visual snapshot. | 00:00 | Central title and diagnostic scale. |
| Full model benchmark | A whole-model measurement across many prompts, layers, wall-clock latency, and memory. | It aggregates runtime and activation behavior across realistic workloads. | This is what would be needed for a broad efficiency claim. | 00:05 | "Not a full benchmark" banner. |
| MFLOPs/token | Millions of floating-point operations for one token in the selected layer. | Dense gated FFN estimate: gate, up, and down projections. | It gives the compute side of the comparison. | 00:10 | Qwen and RWKV metric cards. |
| Delta/MFLOP | Hidden-state movement divided by estimated compute. | Layer delta is normalized by MFLOPs/token. | It is the rough useful-work-per-cost score in this demo. | 00:16 | Highlighted Delta/MFLOP row. |
| Intermediate activations | The actual values inside the expanded FFN or channel-mix layer. | Captured tensors would replace estimated active-neuron and top-k values. | This is needed before turning the demo into a serious efficiency measurement. | 00:22 | Final checklist. |

## Beat Map

| Time | Duration | Spoken lines | Visual action | Key terms | Why the visual change matters |
| --- | ---: | --- | --- | --- | --- |
| 00:00-00:12 | 12s | Final insight: this is a per-layer diagnostic, not a full model benchmark. It measures how much a selected block moves the hidden state relative to its estimated compute. | Title and diagnostic-vs-benchmark scale appear; sampled dot columns fade in behind the title. | Per-layer diagnostic, full model benchmark | Names the correct scope for the metric. |
| 00:12-00:28 | 16s | In this snapshot, Qwen's FFN is about 22 MFLOPs per token, while RWKV's mix is about 14.2 and shows higher Delta per MFLOP. To claim global efficiency, we still need real intermediate activations, latency, and memory. | Qwen/RWKV cards slide in; Delta/MFLOP row highlights; final checklist appears. | MFLOPs/token, Delta/MFLOP, intermediate activations | Gives the measured insight and the missing evidence needed for a stronger conclusion. |

## Voiceover Script

Final insight: this is a per-layer diagnostic, not a full model benchmark. It measures how much a selected block moves the hidden state relative to its estimated compute.

In this snapshot, Qwen's FFN is about 22 MFLOPs per token, while RWKV's mix is about 14.2 and shows higher Delta per MFLOP. To claim global efficiency, we still need real intermediate activations, latency, and memory.
