"""Small dependency-free lexical retriever for local Excel knowledge bases."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Tuple


def _features(text: str) -> Counter[str]:
    clean = re.sub(r"\s+", "", str(text or "").lower())
    chars = [c for c in clean if "\u4e00" <= c <= "\u9fff" or c.isalnum()]
    feats: list[str] = chars[:]
    feats.extend("".join(chars[i : i + 2]) for i in range(max(0, len(chars) - 1)))
    return Counter(feats)


def similarity(left: str, right: str) -> float:
    a, b = _features(left), _features(right)
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return float(dot / (norm_a * norm_b)) if norm_a and norm_b else 0.0


def search(query: str, documents: Iterable[str], top_k: int) -> List[Tuple[int, float]]:
    ranked = sorted(
        ((index, similarity(query, doc)) for index, doc in enumerate(documents)),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[: max(0, top_k)]

