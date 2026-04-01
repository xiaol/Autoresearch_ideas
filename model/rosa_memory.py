from __future__ import annotations

import torch
from torch import nn


class RoSAMemoryStub(nn.Module):
    """Placeholder ROSA module.

    RWKV v8 describes ROSA as a Rapid Online Suffix Automaton.
    This stub returns an embedding of the previous token so the model wiring is intact.
    Replace with a real suffix-automaton implementation later.
    """

    def __init__(self, vocab_size: int, d_model: int, pad_token_id: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pad_token_id = pad_token_id

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T)
        bsz, seqlen = token_ids.shape
        pad = torch.full((bsz, 1), self.pad_token_id, device=token_ids.device, dtype=token_ids.dtype)
        prev = torch.cat([pad, token_ids[:, :-1]], dim=1)
        return self.embed(prev)
