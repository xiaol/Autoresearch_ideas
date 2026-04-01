# LM Engine Integration Notes

LM Engine now has a **runnable UniMatrix integration** in the local workspace, including a shared-depth Universal-Transformer-style model path.

1. **New sequence mixer block**
   - `lm-engine/lm_engine/hf_models/modeling_utils/sequence_mixer_blocks/unimatrix.py`
2. **Registered config + factory hooks**
   - `lm-engine/lm_engine/hf_models/config/sequence_mixer.py`
   - `lm-engine/lm_engine/hf_models/config/__init__.py`
   - `lm-engine/lm_engine/hf_models/modeling_utils/sequence_mixer_blocks/__init__.py`
3. **Cache + dense-model routing**
   - `lm-engine/lm_engine/hf_models/cache/__init__.py`
   - `lm-engine/lm_engine/hf_models/mixins/dense/base.py`
   - `lm-engine/lm_engine/hf_models/mixins/dense/layer.py`
4. **Shared-depth model type**
   - `lm-engine/lm_engine/hf_models/models/unimatrix_ut/config.py`
   - `lm-engine/lm_engine/hf_models/models/unimatrix_ut/base.py`
   - `lm-engine/lm_engine/hf_models/models/unimatrix_ut/main.py`
5. **Smoke tests**
   - `lm-engine/tests/hf_models/single_gpu/unimatrix_test.py`
   - `lm-engine/tests/hf_models/single_gpu/unimatrix_ut_test.py`

## What Is Implemented

- A recurrent **UniMatrix** sequence mixer with:
  - matrix-valued hidden state
  - hybrid outer / diagonal / symmetric update rules
  - optional rule mixing
  - optional state normalization guard
  - recurrent cache support for autoregressive decoding
- FLOP accounting support in:
  - `lm-engine/lm_engine/train_utils.py`
- A **`unimatrix_ut`** model family with:
  - `model_type: unimatrix_ut`
  - shared parameters across depth steps
  - optional step embeddings so the shared block can specialize by recurrence step
  - preserved per-step cache indexing during autoregressive generation

## Verified Locally

- `UniMatrix` forward works with masking and incremental cache updates.
- `UniMatrixUT` instantiates through `AutoModelForCausalLM.from_config(...)`.
- Shared weights are tied across steps for the layer norm, sequence mixer, and MLP blocks.
- Forward pass and `generate(...)` both run successfully with:
  - `PYENV_VERSION=3.13.8`

## Launch Paths

**Apple Silicon training example**

```bash
cd lm-engine
TOKENIZERS_PARALLELISM=false torchrun -m lm_engine.pretrain \
  --config configs/research/unimatrix-ut/apple-silicon.yml
```

**Lightweight throughput benchmark**

```bash
cd lm-engine
PYENV_VERSION=3.13.8 python scripts/mps/benchmark_unimatrix_ut.py \
  --models softmax_attention softmax_attention_shared_depth unimatrix_ut \
  --modes prefill decode_1token forward_backward \
  --seq-lens 128 256 512 1024 \
  --batch-size 4 \
  --timed-repeats 25
```

This benchmark is intentionally conservative:
- it separates prompt prefill, cached one-token decode, and forward-backward timing
- it reports repeated measurements instead of a single timing
- on Apple MPS, memory is a sampled observed allocation, not a CUDA-style exact peak
- the shared-depth attention baseline is a tied-weights proxy, not a separately trained Universal Transformer

## What Is Not Implemented Yet

- Real **ROSA** suffix-automaton memory
- **DeepEmbed** token-conditioned modulation inside LM Engine
- fused kernels or custom scan implementations for high-performance training

So the current LM Engine path should be treated as a **shared-depth UniMatrix core**, not yet the full UniMatrix-ROSA / Discovery stack from the paper repo.
