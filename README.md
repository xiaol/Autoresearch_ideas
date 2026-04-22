# auto_research_llm_ideas

`auto_research_llm_ideas` is primarily a paper workspace: a growing collection of self-contained research-paper packages, supporting benchmark code, and lightweight artifacts used to draft, test, and ship new ideas.

**Current focus:** the `papers/` directory is the center of gravity. Each paper folder is meant to keep the manuscript, key assets, and any handoff-ready package files close together, while the rest of the repo provides the experiment code, notes, and figure-generation infrastructure that feed those papers.

**Status:** active research artifact. The repo now includes a paper library, runnable benchmark scripts, multiple model families, and measured pilot results. It should be read as a working research workshop, not as a broad SOTA claim.

Large generated outputs are intentionally excluded from git by default:
- experiment dumps under `results/`
- rendered media under `video/renders/` and `video/voiceover/`

The repo is meant to ship reproducible research code, paper sources, and lightweight assets. Heavy outputs can be regenerated locally from the included scripts.

## Papers First

The main deliverable in this repo is the paper collection under `papers/`. These folders are the easiest way to browse the research output directly.

**Current paper packages**
- `papers/persistent_compiled_knowledge_agent_memory/` - *Persistent Compiled Knowledge as Agent Memory*, a paper on Andrej Karpathy's LLM Wiki pattern as durable agent memory infrastructure, with PaperX analysis assets and `main.pdf`.
- `papers/associative_state_universal_transformers/` - *Associative-State Universal Transformers*, the main UniMatrix manuscript package with self-contained LaTeX sources and figures.
- `papers/generic_triple_latent/` - *Generic Triple-Latent Compression with Gated Associative Retrieval*, a standalone paper package with finalized paper artifacts and supporting figure/code assets.
- `papers/recurrent_ffn/` - *Replacing Transformer FFNs with Localized Recurrent Memory*, a standalone recurrent-memory paper draft with local bibliography and build files.
- `papers/arc_agi2_grid_native/` - *Short-Long State Memory for Sub-Million Grid-Native ARC-AGI-2 Reasoning*, including the paper draft plus arXiv submission materials.
- `papers/on_device_meta_learning_agents/` - *Decoupling Learning from Inference: The Rise of On-Device Meta-Learning Paradigms for Agents*, with paper source markdown, LaTeX, and PDF artifact.
- `papers/on_policy_self_distillation/` - *GOLD-HAM: On-Policy Distillation for Selective Storage in Hybrid Associative Memory Architectures*, with source PDFs, figures, and compiled manuscript.
- `papers/lpm_performance_model/` - *LPM 1.0: End-to-End Real-Time Video Character Performance*, with the manuscript plus a large set of extracted page and figure assets.
- `papers/reward_hacking/` - *Reward Hacking in Agentic AI Systems: From Vulnerabilities to Architectural Solutions*, with the LaTeX source and compiled paper artifact.
- `papers/rwkv_concurrent_rollout/` - *Concurrent Rollout Search with Fixed-Size RWKV State*, with compiled manuscript and rendered page previews.
- `papers/jepa_rosa/` - JEPA-ROSA starter workspace with concept framing and experiment plan for a future paper package.
- `papers/generic_triple_latent_arxiv/` - arXiv-oriented export package for the generic triple-latent paper.

If you only read one part of the repo, start with `papers/` and then follow references from each package into `docs/`, `experiments/`, `results/`, or `assets/` as needed.

**Repo Map**
- `papers/` – primary paper library with manuscript packages, handoff artifacts, and research-paper snapshots
- `docs/` – literature summaries, architecture notes, benchmark audits, and paper-planning documents
- `experiments/` – runnable language-model, reasoning, memory, and throughput benchmarks that feed the papers
- `model/` – PyTorch model implementations and research baselines
- `configs/` – example training and evaluation configs
- `results/` – local experiment outputs; large generated files are excluded from git by default
- `video/` – script, storyboard, and production code for video explainers tied to selected papers
- `references/` – citation links and bibliography material

## Book and Skill Package

This repo now also includes a full book package:

- `books/machine-learning-full-course-by-ai/` – the complete mdBook source, built HTML, review PDF, runnable examples, and course-production workspace

The book is not only a manuscript. It is paired with real local reader skills:

- `.agents/skills/` – reusable ML skills such as `ml-math-translator`, `ml-problem-framer`, `ml-baseline-builder`, `ml-evaluation-review`, `ml-retrieval-evaluation`, and related review skills

Good starting points:

- `books/machine-learning-full-course-by-ai/README.md`
- `books/machine-learning-full-course-by-ai/book/index.html`
- `books/machine-learning-full-course-by-ai/src/how-to-use-reader-skills.md`
- `books/machine-learning-full-course-by-ai/src/appendix-b.md`

The intended workflow is:

1. Read a chapter.
2. Run the matching example or inspect the chapter case.
3. Invoke the matching reader skill.
4. Save the result as a reusable memo, review, or checklist.

That package is meant to make the repo useful not only for paper-writing and experiments, but also for AI-native machine learning learning and teaching.

## LM Engine Next Move

A runnable `unimatrix` + `unimatrix_ut` path is now wired into the local [`/Users/xiaol/x/PaperX/lm-engine`](/Users/xiaol/x/PaperX/lm-engine) workspace. The current integration covers the recurrent core, shared-depth tied-weight execution, recurrent caching, and generation. Real ROSA memory, DeepEmbed modulation, and fused kernels are still future work.

**LM Engine quick launch**
```bash
cd lm-engine
TOKENIZERS_PARALLELISM=false torchrun -m lm_engine.pretrain \
  --config configs/research/unimatrix-ut/apple-silicon.yml
```

**LM Engine benchmark helper**
```bash
cd lm-engine
PYENV_VERSION=3.13.8 python scripts/mps/benchmark_unimatrix_ut.py \
  --models softmax_attention softmax_attention_shared_depth unimatrix_ut \
  --modes prefill decode_1token forward_backward \
  --seq-lens 128 256 512 1024 \
  --batch-size 4 \
  --timed-repeats 25
```

The benchmark helper is meant to be a stricter reference benchmark, not a production serving claim. It separates prefill, one-token decode, and forward-backward timing, and it labels the MPS memory metric conservatively.

**Quick Start (toy model)**
```bash
python - <<'PY'
import torch
from model.unimatrix_rosa import ModelConfig, UniMatrixRosaLM

cfg = ModelConfig(vocab_size=32000, d_model=512, n_layers=6, n_heads=4, state_dim=64, deep_embed_dim=256)
model = UniMatrixRosaLM(cfg)
tokens = torch.randint(0, cfg.vocab_size, (2, 32))
logits, _ = model(tokens)
print(logits.shape)
PY
```

If you want this wired into `lm-engine`, see `docs/02_architecture_proposal.md`, `docs/03_training_eval_plan.md`, and `docs/05_lm_engine_integration.md` for the current implementation status and launch path.

## RULER Core Eval

A supported `RULER`-style long-context subset is now wired into the repo for the self-contained synthetic tasks:
- `niah_single_1`
- `vt`
- `cwe`
- `fwe`

This path is intentionally conservative:
- it matches official RULER-style task names and string-match scoring for the supported tasks
- it excludes QA and essay-based haystacks, which depend on external corpora/download steps
- it is best treated as a `RULER core subset`, not the full official leaderboard pipeline

Run it from a local trained checkpoint:

```bash
python -m auto_research_llm_ideas.experiments.ruler_subset \
  --checkpoint auto_research_llm_ideas/results/pilot_neurips/lm/unimatrix-core_lm.pt \
  --tasks niah_single_1 vt cwe fwe \
  --num-samples 32 \
  --max-seq-length 128 \
  --output-dir auto_research_llm_ideas/results/ruler_core
```

Or run it through the full suite after training:

```bash
python -m auto_research_llm_ideas.experiments.run_full_suite \
  --output-root auto_research_llm_ideas/results/pilot_neurips \
  --models transformer unimatrix-core unimatrix-rosa unimatrix-discovery \
  --lm-steps 80 \
  --memory-steps 200 \
  --bench-seq-lens 64 128 256 512 \
  --d-model 64 \
  --n-layers 3 \
  --n-heads 4 \
  --state-dim 16 \
  --ruler-tasks niah_single_1 vt cwe fwe \
  --ruler-samples 8 \
  --ruler-max-seq-length 128
```

## Pilot Experiments

Run the full pilot suite:

```bash
python -m auto_research_llm_ideas.experiments.run_full_suite \
  --output-root auto_research_llm_ideas/results/pilot_neurips \
  --models transformer unimatrix-core unimatrix-rosa unimatrix-discovery \
  --lm-steps 80 \
  --memory-steps 200 \
  --bench-seq-lens 64 128 256 512 \
  --d-model 64 \
  --n-layers 3 \
  --n-heads 4 \
  --state-dim 16
```

Run the higher-order triple-interaction suite:

```bash
python -m auto_research_llm_ideas.experiments.run_higher_order_suite \
  --output-root auto_research_llm_ideas/results/higher_order \
  --models transformer-triple exact-triplet pair-state pair-slot hybrid-pair typed-latent \
  --steps 300 \
  --batch-size 32 \
  --seq-len 65 \
  --bench-seq-lens 32 48 64 80 \
  --d-model 128 \
  --n-layers 3 \
  --n-heads 4 \
  --state-dim 16 \
  --num-slots 8 \
  --train-tasks binding-copy binding-affine binding-gate binding-lookup \
  --eval-tasks binding-copy binding-affine binding-gate binding-lookup
```

The higher-order suite now writes explicit baseline comparisons against standard attention:
- `training_vs_standard_attention.csv`
- `benchmark_vs_standard_attention.csv`

That lets you compare each triple-interaction variant directly to `transformer-triple` on:
- task accuracy deltas
- loss deltas
- parameter ratio
- train-time ratio
- throughput, latency, and memory ratios

Important benchmark fix:
- the higher-order benchmark now prepends an explicit task token
- earlier multi-task runs without a task token were ambiguous, because the same sequence could map to different labels under `copy`, `affine`, `gate`, and `lookup`
- mixed-curriculum comparisons should therefore use the task-prompted benchmark only

For a clean apples-to-apples comparison to standard attention, keep the following fixed:
- tokenization and synthetic task generator
- `seq_len`, `batch_size`, `steps`, and optimizer settings
- model width/depth when you want raw architectural comparison

Then also report a second regime with matched parameter budget, since true triple-interaction models can trade parameters for compute very differently.

Run the staged curriculum suggested by the pilot:

```bash
python -m auto_research_llm_ideas.experiments.run_higher_order_staged \
  --output-root auto_research_llm_ideas/results/higher_order_staged \
  --models transformer-triple typed-latent hybrid-pair \
  --stage-tasks binding-copy binding-affine binding-gate binding-lookup \
  --stage-steps 120 \
  --batch-size 16 \
  --seq-len 49 \
  --bench-seq-lens 16 24 32 \
  --d-model 64 \
  --n-layers 2 \
  --n-heads 4 \
  --state-dim 8 \
  --num-slots 4 \
  --aux-loss-weight 0.5
```

This staged runner:
- trains one task at a time with checkpointed continuation
- saves per-stage checkpoints under `training/<model>/aux_<weight>/checkpoints/`
- writes `final_results.csv` plus `final_vs_standard_attention.csv`
- supports `--resume` to continue from the latest completed stage

The higher-order suite compares:
- `transformer-triple`: standard attention baseline
- `exact-triplet`: explicit cubic three-token interactions
- `pair-state`: compressed pair-memory latent for true triple effects
- `pair-slot`: low-rank slot compression of pair memory
- `hybrid-pair`: local mixing plus pair-memory recurrence
- `typed-latent`: benchmark-aware typed latent compression with exact symbolic heads for `copy/affine/gate` and a learned compressed lookup table

The task ladder is:
- `binding-copy`: retrieve the `A` value associated with the queried tag
- `binding-affine`: compose the queried tag's `A/B/C` values with a fixed arithmetic rule
- `binding-gate`: compose the queried tag's values with a non-linear gated rule
- `binding-lookup`: compose the queried tag's values with a random lookup table

Current strongest corrected higher-order result:
- staged `typed-latent` reaches `100%` accuracy on all four tasks in `results/_staged_higher_order_typedlatent_v2`
- the same staged run leaves `transformer-triple` at `12.5% / 6.1% / 7.4% / 9.5%` on `copy / affine / gate / lookup`
- this should be interpreted as an internal benchmark SOTA and a constructive proof that latent state can compress true triple-token interactions, not as a generic language-model SOTA claim

Run the generic latent follow-up:

```bash
python -m auto_research_llm_ideas.experiments.run_generic_latent_followup \
  --output-root auto_research_llm_ideas/results/generic_latent_followup \
  --lm-steps 80 \
  --memory-steps 200 \
  --bench-seq-lens 64 128 256 512 \
  --d-model 64 \
  --n-layers 3 \
  --n-heads 4 \
  --state-dim 16
```

This follow-up separates two questions:
- does latent compression help on a corrected mechanism benchmark with exposed structure
- does a generic, untyped latent compressor help on a standard language-model benchmark

The runner writes two regimes:
- `matched_width/`: same `d_model=64` width for Transformer and latent variants
- `param_matched_lm/`: approximate parameter matching for LM only (`triple-latent` and `triple-slot` at `d_model=60`, `triple-hybrid` at `d_model=56`)

Measured pilot outputs from the current run are in:
- `results/_generic_latent_v1/`
- `results/_generic_latent_parammatched_v1/`
- `results/_triple_latent_budget_sweep_v1/`
- `results/_winner_vs_transformer_seeds_v1/`
- `results/_winner_triple_hybrid_d64s8/`

Current generic latent result:
- matched-width `triple-hybrid` reaches `4.766` BPB on byte-level WikiText-2 versus `5.124` for the Transformer baseline
- parameter-matched `triple-hybrid` still reaches `5.067` BPB at `169,424` params versus `174,848` for the Transformer
- a near-budget sweep finds a stronger winner: `triple-hybrid` with `d_model=64`, `state_dim=8` reaches `4.827` BPB at `177,752` params
- the same near-budget winner averages `4.806` BPB across three seeds, versus `5.143` for the Transformer across the same seeds
- all three generic latent variants remain at `13.4%` associative-recall accuracy versus `25.4%` for the Transformer
- absolute throughput is still far worse in the current reference implementation, although the latent variants stay nearly flat as sequence length increases

Compile the main paper:

```bash
cd auto_research_llm_ideas/papers/associative_state_universal_transformers
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Current Pilot Result

- `unimatrix-rosa` reaches `5.083` validation bits-per-byte on byte-level WikiText-2 versus `5.124` for the Transformer baseline, using `40.6%` fewer parameters.
- `unimatrix-core` is almost identical in quality and uses `52.5%` fewer parameters.
- The current UniMatrix variants still trail the Transformer on associative recall, so the long-range routing story is not finished yet.

## Architecture Variants

See the PNG gallery and per-variant notes:
- `docs/06_architecture_gallery.md`
- `docs/architectures/01_umt_dynamic.md`
- `docs/architectures/02_umt_rosa.md`
- `docs/architectures/03_umt_deepembed.md`
- `docs/architectures/04_umt_structured.md`
- `docs/architectures/05_umt_hybrid.md`
- `docs/architectures/06_umt_dual_timescale.md`
- `docs/architectures/07_umt_rulemix.md`
- `docs/architectures/08_umt_skewstable.md`
- `docs/architectures/09_umt_convmix.md`
- `docs/architectures/10_umt_stepconditioned.md`
- `docs/architectures/11_umt_spectral.md`
- `docs/architectures/12_umt_discovery.md`
