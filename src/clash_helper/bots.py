"""The ladder bots' preset decks, collected by research/bots.py.

Most opponents near 4300 trophies are bots that show a clan name the API does not know, so their
deck cannot be looked up. They play a fixed set of decks though, and a few of their cards give
away the rest.
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path

PATH = Path(__file__).parent / "data" / "bot_decks.json"
SHOWN = 3  # cards they must have shown before a preset deck is trusted
SURE = 0.6  # share of the bot games with those cards one deck must account for


@cache
def decks() -> list[tuple[frozenset[str], int]]:
    path = Path(os.environ.get("CLASH_BOT_DECKS", PATH))
    return [(frozenset(d), n) for d, n in json.loads(path.read_text())] if path.exists() else []


def deck(shown: frozenset[str]) -> list[str] | None:
    """The preset deck that holds the cards they have shown once one clearly leads, else None. One
    shown card may be missing from it, a misread or one of your own units taken for theirs."""
    if len(shown) < SHOWN:
        return None
    # a stray card is only forgiven once enough others agree
    fits = [(k, d, n) for d, n in decks() for k in [len(shown & d)]
            if k == len(shown) >= SHOWN or k == len(shown) - 1 >= SHOWN + 1]
    if not fits:
        return None
    most = max(k for k, _, _ in fits)
    top, n = max(((d, n) for k, d, n in fits if k == most), key=lambda m: m[1])
    return sorted(top) if n >= SURE * sum(n for k, _, n in fits if k == most) else None
