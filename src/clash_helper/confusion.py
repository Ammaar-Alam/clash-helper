"""What a detector class really means, learned from labeled recordings.

The detector names units after the classes it was trained on, so newer cards come out as
look-alikes (Vines as Poison) and some effects come out as spells that were never cast.
data/confusion.json counts, for every detector class, which true card it turned out to be
or "none" when it was no play at all. research/train.py rebuilds it from the labels.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from functools import cache
from pathlib import Path

NONE = "none"
PATH = Path(__file__).parent / "data" / "confusion.json"
TRUST = 3.0  # pseudo counts that the detector means what it says, for classes with little data
NONE_PRIOR = 0.3  # pseudo counts that any candidate was no play
FLOOR = 0.02


@cache
def table() -> dict[str, dict[str, float]]:
    path = Path(os.environ.get("CLASH_CONFUSION", PATH))
    return json.loads(path.read_text()) if path.exists() else {}


def meaning(detected: str, candidates: list[str]) -> dict[str, float]:
    """P(true card or none | detector said `detected`), over the candidates.

    Whether it was a play at all and which card it was are learned separately, so a class
    that is never in this deck still counts as some play rather than as nothing.
    """
    learned = Counter(table().get(detected, {}))
    none = (learned[NONE] + NONE_PRIOR) / (sum(learned.values()) + NONE_PRIOR + TRUST)
    weights = {c: learned[c] + (TRUST if c == detected else 0.0) + FLOOR for c in candidates}
    total = sum(weights.values()) or 1.0
    return {**{c: (1 - none) * w / total for c, w in weights.items()}, NONE: none}


def emission(votes: Counter, candidates: list[str]) -> dict[str, float]:
    """Log likelihood of each candidate card and of no play, given the detector's votes."""
    total = sum(votes.values())
    if not total:
        return {c: 0.0 for c in [*candidates, NONE]}
    mix: Counter = Counter()
    for detected, weight in votes.items():
        for c, p in meaning(detected, candidates).items():
            mix[c] += weight / total * p
    return {c: math.log(max(mix[c], 1e-6)) for c in [*candidates, NONE]}
