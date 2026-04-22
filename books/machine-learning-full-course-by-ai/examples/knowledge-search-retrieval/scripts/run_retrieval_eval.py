#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = ROOT / "data" / "documents.csv"
QUERIES_PATH = ROOT / "data" / "queries.csv"


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_documents() -> list[dict[str, str]]:
    with DOCS_PATH.open() as f:
        return list(csv.DictReader(f))


def load_queries() -> list[dict[str, str]]:
    with QUERIES_PATH.open() as f:
        return list(csv.DictReader(f))


def tf(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values()) or 1
    return {k: v / total for k, v in counter.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in set(a) | set(b))
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> None:
    docs = load_documents()
    queries = load_queries()
    doc_vectors = []
    for doc in docs:
        text = f"{doc['title']} {doc['text']}"
        doc_vectors.append((doc, tf(Counter(tokenize(text)))))

    correct = 0
    print("Knowledge Search Retrieval Example")
    print()
    for query in queries:
        q_vec = tf(Counter(tokenize(query["query"])))
        scored = []
        for doc, vec in doc_vectors:
            scored.append((cosine(q_vec, vec), doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_doc = scored[0][1]
        ok = top_doc["doc_id"] == query["expected_doc_id"]
        if ok:
            correct += 1
        print(f"Query: {query['query']}")
        print(f"Expected doc: {query['expected_doc_id']}")
        print(f"Top hit: {top_doc['doc_id']} | {top_doc['title']} | ok={ok}")
        print()

    accuracy = correct / len(queries)
    print(f"Top-1 accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
