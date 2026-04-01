# Results Excluded From Git

This repo keeps experiment code, paper sources, and selected paper figures in
git, but excludes large generated experiment outputs by default.

Typical local result roots include:

- `results/pilot_neurips/`
- `results/sparsepointer_ablations_d0/`
- `results/_winner_vs_transformer_seeds_v1/`
- `results/recurrent_ffn_tuned_compare_v1/`

Reproduce them with the scripts in:

- `experiments/run_full_suite.py`
- `experiments/run_sparsepointer_ablations.py`
- `experiments/run_generic_latent_followup.py`
- `experiments/run_higher_order_staged.py`
- `experiments/run_long_context_transfer.py`
