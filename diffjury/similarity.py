"""Dependency-free lexical similarity for knowledge admission and retrieval.

Knowledge items are short, code-flavoured sentences ("Parameterize SQL built
with f-strings in cursor.execute"). A TF-IDF cosine over identifier-aware
tokens is cheap, deterministic and good enough to shortlist near-duplicates
before an LLM judge decides whether they are really the same lesson.
"""
import hashlib
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+|[一-鿿]")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOPWORDS = frozenset("""
a an and are as at be been by can do does for from has have if in into is it its
may must not of on or should so such than that the then there these this those to
use used uses using via was were when where which while will with without you your
""".split())


def tokenize(text: str) -> List[str]:
    """Split prose and identifiers (snake_case, camelCase, CWE-89) into terms."""
    tokens: List[str] = []
    for word in _WORD.findall(str(text or "")):
        parts = _CAMEL.findall(word) if not word.isdigit() else [word]
        for part in parts or [word]:
            value = part.lower()
            if value.isdigit() or (
                value not in STOPWORDS and (len(value) > 1 or not value.isascii())
            ):
                tokens.append(value)
    return tokens


def fingerprint(text: str) -> str:
    """Order-insensitive identity of a sentence's content words."""
    normalized = " ".join(sorted(set(tokenize(text))))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


class TfidfIndex:
    """Small in-memory TF-IDF index rebuilt per admission or assembly call."""

    def __init__(self, documents: Dict[str, str]):
        self.ids = list(documents)
        counts = {key: Counter(tokenize(value)) for key, value in documents.items()}
        frequency: Counter = Counter()
        for terms in counts.values():
            frequency.update(terms.keys())
        total = max(1, len(counts))
        self.idf = {
            term: math.log((1 + total) / (1 + df)) + 1.0
            for term, df in frequency.items()
        }
        self.vectors = {key: self._weigh(terms) for key, terms in counts.items()}

    def _weigh(self, terms: Counter) -> Dict[str, float]:
        vector = {
            term: (1.0 + math.log(count)) * self.idf.get(term, 1.0)
            for term, count in terms.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {term: value / norm for term, value in vector.items()}

    def query(self, text: str) -> Dict[str, float]:
        return self._weigh(Counter(tokenize(text)))

    def similarity(self, text: str, key: str) -> float:
        return cosine(self.query(text), self.vectors.get(key, {}))

    def top_k(self, text: str, k: int = 5, exclude: Iterable[str] = ()) -> List[Tuple[str, float]]:
        vector = self.query(text)
        skipped = set(exclude)
        scored = [
            (key, cosine(vector, self.vectors[key]))
            for key in self.ids if key not in skipped
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [item for item in scored[:max(0, k)] if item[1] > 0.0]


def cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return round(sum(value * right.get(term, 0.0) for term, value in left.items()), 6)


def pairwise_duplicates(
    documents: Dict[str, str], threshold: float,
) -> List[Tuple[str, str, float]]:
    """Return every pair whose cosine is at least ``threshold`` (highest first)."""
    index = TfidfIndex(documents)
    pairs = []
    ids: Sequence[str] = index.ids
    for position, left in enumerate(ids):
        for right in ids[position + 1:]:
            score = cosine(index.vectors[left], index.vectors[right])
            if score >= threshold:
                pairs.append((left, right, score))
    pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
    return pairs
