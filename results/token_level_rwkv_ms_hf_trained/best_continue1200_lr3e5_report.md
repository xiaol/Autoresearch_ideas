# RWKV-MS HF Continuation Result

Best adapter after this improvement pass:

`/run/media/xiaol/B214449214445C0B/delta_mem_outputs/gemma_rwkv_ms_hf_mix/raw_lm_5mchars_from800_continue1200_len256_lr3e5_20260702_085744`

## Training

Started from the previous 800-step HF-trained adapter and continued for 1200 more
steps at learning rate `3e-5`, `max_length=256`, batch size 1.

Held-out trainer validation loss:

| Step | Validation loss |
| ---: | ---: |
| Previous adapter final | 6.080515 |
| 200 | 6.043161 |
| 400 | 6.024363 |
| 600 | 6.005088 |
| 800 | 5.990652 |
| 1000 | 5.981460 |
| 1200 | 5.969240 |

## Full Validation Token Eval

Validation set: prepared HF mix validation JSONL, 41 records, 32,244 scored
tokens. Positive gap means RWKV-MS assigned higher probability than the base
model.

| Adapter | Base NLL | RWKV-MS NLL | Gap | Positive-token fraction |
| --- | ---: | ---: | ---: | ---: |
| Previous 800-step adapter | 6.418264 | 5.789028 | 0.629236 | 0.561221 |
| Continued 1200-step adapter | 6.418264 | 5.685024 | 0.733240 | 0.570711 |

Domain gaps for the continued adapter:

| Domain | Tokens | Gap | Positive-token fraction |
| --- | ---: | ---: | ---: |
| html | 4,894 | 0.347529 | 0.530854 |
| latex | 3,984 | 0.887516 | 0.569026 |
| prose | 16,289 | 0.877949 | 0.586408 |
| python | 7,077 | 0.580050 | 0.563092 |

Filtered validation gaps:

| Filter | Count | Gap |
| --- | ---: | ---: |
| all_tokens | 32,244 | 0.733240 |
| top_10_open_class_no_copy_4 | 3,989 | 1.382450 |
| copy_5_only | 4,855 | 0.245136 |

## Pilot And Synthetic Checks

Small four-document pilot all-token gap improved from `-0.115173` to `-0.002499`.
Open-class non-copy pilot gap improved from `0.190217` to `0.340361`.

Synthetic probes remain mixed. Pronoun margins improved at 32 and 128, entity
accuracy stayed the same as the previous adapter, and structural closure remains
positive at distances 32 and 128 while distance 64 moved from slightly negative
to approximately neutral.

## Artifacts

- Full validation summary:
  `results/token_level_rwkv_ms_hf_trained/validation_eval_continue1200_lr3e5/summary/filtered_losses.json`
- Pilot summary:
  `results/token_level_rwkv_ms_hf_trained/pilot_continue1200_lr3e5/summary/filtered_losses.json`
- Synthetic summary:
  `results/token_level_rwkv_ms_hf_trained/synthetic_continue1200_lr3e5/synthetic_summary.csv`
