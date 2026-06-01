from __future__ import annotations

from importlib import resources
from pathlib import Path


class RWKVTokenizer:
    """RWKV world tokenizer, copied from the upstream inference demos."""

    def __init__(self, file_name: str | Path | None = None) -> None:
        self.idx2token: dict[int, bytes] = {}
        sorted_tokens: list[bytes] = []

        if file_name is None:
            with resources.files(__package__).joinpath(
                "rwkv_vocab_v20230424.txt"
            ).open("r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = Path(file_name).read_text(encoding="utf-8").splitlines()

        for line in lines:
            idx = int(line[: line.index(" ")])
            raw = eval(line[line.index(" ") : line.rindex(" ")])
            token = raw.encode("utf-8") if isinstance(raw, str) else raw
            if not isinstance(token, bytes):
                raise TypeError(f"unexpected token type at index {idx}: {type(token)}")
            if len(token) != int(line[line.rindex(" ") :]):
                raise ValueError(f"bad token length at index {idx}")
            sorted_tokens.append(token)
            self.idx2token[idx] = token

        self.token2idx = {value: int(key) for key, value in self.idx2token.items()}

        self.table: list[list[list[bytes]]] = [[[] for _ in range(256)] for _ in range(256)]
        self.good: list[set[int]] = [set() for _ in range(256)]
        self.wlen: list[int] = [0 for _ in range(256)]

        for token in reversed(sorted_tokens):
            if len(token) >= 2:
                first = int(token[0])
                second = int(token[1])
                self.table[first][second].append(token)
                self.wlen[first] = max(self.wlen[first], len(token))
                self.good[first].add(second)

    def encode_bytes(self, src: bytes) -> list[int]:
        src_len = len(src)
        tokens: list[int] = []
        i = 0
        while i < src_len:
            token = src[i : i + 1]
            if i < src_len - 1:
                second = int(src[i + 1])
                first = int(src[i])
                if second in self.good[first]:
                    prefix = src[i : i + self.wlen[first]]
                    try:
                        token = next(filter(prefix.startswith, self.table[first][second]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[token])
            i += len(token)
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        return b"".join(self.idx2token[int(i)] for i in tokens)

    def encode(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")
