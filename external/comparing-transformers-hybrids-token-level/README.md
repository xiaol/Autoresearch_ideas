# Comparing Transformers and Hybrid Models at the Token Level

Implementation scaffold for arXiv:2606.20936, "Comparing Transformers and Hybrid
Models at the Token Level".

The paper is an evaluation method, not a new trainable architecture. This package
implements the main diagnostics:

- paired token-level NLL comparison on the same prefixes and target tokens
- source-level tagging for prose, Python, HTML, and LaTeX-like text
- character-span alignment from source tags to LM target tokens
- raw tag summaries, repeated n-gram/copy filters, and lightweight OLS controls
- synthetic pronoun-memory, entity-tracking, and structural-closure probes

The implementation is model-agnostic. The paper uses matched OLMo checkpoints,
but this repo's local path should use the Multi-state RWKV online-memory adapter:
frozen Gemma4 E4B as the baseline and Gemma4 E4B + RWKV-MS delta-Mem as the
memory model.

## Install

```bash
cd external/comparing-transformers-hybrids-token-level
python -m pip install -e .
```

For POS tagging, install NLTK data if you want better prose tags:

```bash
python - <<'PY'
import nltk
nltk.download("averaged_perceptron_tagger_eng")
PY
```

Without NLTK data the prose tagger falls back to deterministic regex heuristics.

## Local RWKV-MS Natural-token scoring

Use this for the local Multi-state RWKV online-memory version:

```bash
python -m token_level_eval.score_rwkv_ms \
  --delta-mem-root /home/xiaol/X/delta-Mem \
  --base-model google/gemma-4-E4B-it \
  --memory-dir /home/xiaol/X/hf_gemma_rwkv_step100_upload \
  --input data/prose_sample.txt \
  --domain prose \
  --seq-len 8192 \
  --output-jsonl results/rwkv_ms_token_rows.jsonl
```

The output schema keeps the paper's paired-loss names for compatibility:

- `loss_transformer` is the frozen base-model NLL
- `loss_hybrid` is the base + RWKV-MS online-memory NLL
- `loss_gap = loss_transformer - loss_hybrid`, so positive means RWKV-MS assigned
  higher probability to the observed target

The RWKV-MS scorer resets online memory at every packed evaluation window, so a
row's target is conditioned on the same local prefix window as the baseline.

## Generic Hugging Face Pair Scoring

Prepare a text file, a directory of `.txt` files, or JSONL rows with a text field.

```bash
python -m token_level_eval.score \
  --transformer-model HF_BASELINE_MODEL_OR_PATH \
  --hybrid-model HF_MEMORY_OR_HYBRID_MODEL_OR_PATH \
  --input data/prose_sample.txt \
  --domain prose \
  --seq-len 8192 \
  --output-jsonl results/token_rows.jsonl
```

Use `--jsonl-text-key text` when the input is JSONL. Use `--domain auto` for a
simple suffix-based domain guess.

Each output row is one scored target token with:

- `loss_transformer`, `loss_hybrid`, and `loss_gap`
- source `tags` and `fine_tags`
- `word_position`, `rel_pos`, `copy_1` ... `copy_N`, and `prev_distance`

`loss_gap = loss_transformer - loss_hybrid`, so positive means the hybrid assigned
higher probability to the observed target.

## Summaries, filters, and controls

```bash
python -m token_level_eval.summarize \
  --input-jsonl results/token_rows.jsonl \
  --output-dir results/token_summary \
  --max-copy-ngram 16 \
  --run-regression
```

Outputs include:

- `tag_summary.csv`: raw tag-stratified gaps
- `copy_summary.csv`: repeated n-gram gaps
- `filtered_losses.json`: all-token, top-open-class/no-copy, and copy-only filters
- `regression_coefficients.csv`: lightweight OLS controls for tags, domain, word
  position, difficulty, frequency, relative position, previous distance, and copy
  features

The regression is a practical diagnostic, not an exact recreation of every
plotting choice in the paper.

## Local RWKV-MS Synthetic probes

```bash
python -m token_level_eval.synthetic_rwkv_ms \
  --delta-mem-root /home/xiaol/X/delta-Mem \
  --base-model google/gemma-4-E4B-it \
  --memory-dir /home/xiaol/X/hf_gemma_rwkv_step100_upload \
  --distances 32 64 128 256 512 1024 \
  --num-examples 100 \
  --output-dir results/rwkv_ms_synthetic_probes
```

## Hugging Face training mix for RWKV-MS adapter experiments

Use the dataset prep script when training a broader RWKV-MS adapter before
rerunning the paper-style diagnostics. It builds a document-disjoint
train/validation JSONL mix from public Hugging Face datasets:

- `wikimedia/wikipedia` (`20231101.en`) and `vblagoje/cc_news` for prose
- `ccdv/arxiv-summarization` for academic prose
- `code-search-net/code_search_net` (`python`) for Python
- `bigcode/the-stack-smol-xl` language files for HTML and Markdown
- `scholarweave/arxiv-latex` for LaTeX source

The output rows use the same `id`, `domain`, and `text` shape as the token-level
scorers, plus source metadata. Keep the output on the mounted SSD rather than
the nearly full root filesystem.

```bash
python scripts/prepare_rwkv_ms_hf_dataset.py \
  --output-dir /run/media/xiaol/B214449214445C0B/autoresearch_datasets/rwkv_ms_hf_mix_5mchars \
  --hf-cache /run/media/xiaol/B214449214445C0B/hf_cache \
  --target-train-chars 5000000
```

For a first real adapter-training run, scale the same recipe to roughly
50M tokens by using about 200M characters:

```bash
python scripts/prepare_rwkv_ms_hf_dataset.py \
  --output-dir /run/media/xiaol/B214449214445C0B/autoresearch_datasets/rwkv_ms_hf_mix_200mchars \
  --hf-cache /run/media/xiaol/B214449214445C0B/hf_cache \
  --target-train-chars 200000000
```

Train a Gemma4 + RWKV-MS adapter from that JSONL using raw next-token LM windows:

```bash
PYTHONPATH=src:/home/xiaol/X/delta-Mem python -m token_level_eval.train_rwkv_ms_lm \
  --delta-mem-root /home/xiaol/X/delta-Mem \
  --base-model /run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it \
  --train-jsonl /run/media/xiaol/B214449214445C0B/autoresearch_datasets/rwkv_ms_hf_mix_5mchars/train.jsonl \
  --validation-jsonl /run/media/xiaol/B214449214445C0B/autoresearch_datasets/rwkv_ms_hf_mix_5mchars/validation.jsonl \
  --output-dir /run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_hf_mix/raw_lm_5mchars_step400 \
  --resume-adapter-dir /home/xiaol/X/hf_gemma_rwkv_step100_upload \
  --max-length 256 \
  --max-steps 400
```

## Generic Hugging Face Synthetic probes

```bash
python -m token_level_eval.synthetic \
  --models transformer=HF_TRANSFORMER_MODEL_OR_PATH hybrid=HF_HYBRID_MODEL_OR_PATH \
  --distances 32 64 128 256 512 1024 \
  --num-examples 100 \
  --output-dir results/synthetic_probes
```

Pronoun memory and entity tracking are contrastive:

```text
margin = log p(correct | prefix) - log p(distractor | prefix)
accuracy = 1[margin > 0]
```

Structural closure reports NLL on the closing token.
