"""Name a whole sequence of enemy plays at once using the card cycle.

A card can only be played from the hand, and the hand is the deck minus the
last 4 plays. So among any 4 plays in a row no card repeats. Each play has
weak evidence on its own, the cycle ties them together, and a beam search
finds the most likely sequence. Breaking the rule is allowed at a cost
because some plays are never seen. A candidate can also turn out to be no
play at all, a spawn or an effect the detector mistook for a spell.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from clash_helper import confusion
from clash_helper.cards import all_cards, card

NONE = confusion.NONE
BEAM = 300
REPEAT_COST = 3.0  # nats, a repeat within 4 plays needs strong evidence
COUNT_WEIGHT = 0.5  # per unit of log ratio between the units a card brings and those seen at once
# how believable a candidate is before looking at what the detector saw, by how it was found
NONE_BIAS = {"timer": -2.0, "bolt": -2.0, "birth": -0.5, "spell": 0.0}


@dataclass
class Evidence:
    votes: Counter  # card -> summed detector confidence
    units: int  # most units seen landing at once
    digits: int | None  # digits in the level badge
    named: str | None = None
    spell: bool = False  # seen as a spell rather than a unit landing
    track: int | None = None  # the detector track that showed it, its later frames refine the name
    source: str = "timer"  # timer, birth, bolt or spell


def scores(evidence: Evidence, deck: list[str] | None, levels: dict[str, int]) -> dict[str, float]:
    """Log score for each possible card and for no play at all."""
    pool = deck if deck is not None else list(all_cards())
    # a spell never lands with a timer and a unit never flies in like a spell
    kind = [c for c in pool if card(c).spell == evidence.spell]
    out = confusion.emission(evidence.votes, kind, evidence.spell)
    for name in kind:
        count = card(name).count
        if evidence.units and count:
            # fewer units than the card brings is common, they hide and their boxes merge. Clearly more
            # at once is not, a doubled box aside
            extra = evidence.units - count - 1
            out[name] -= extra if extra > 0 else COUNT_WEIGHT * math.log(max(1.0, count / evidence.units))
        if evidence.digits and name in levels:
            out[name] += 0.5 if len(str(levels[name])) == evidence.digits else -0.5
    out[NONE] += NONE_BIAS.get(evidence.source, 0.0)
    return out


def best(evidence: Evidence, deck: list[str] | None = None) -> str:
    """The single most likely reading of one candidate, possibly NONE."""
    s = scores(evidence, deck, {})
    return max(s, key=s.get)


def decode(evidence: list[Evidence], deck: list[str], levels: dict[str, int]) -> list[str]:
    """Most likely card, or NONE, for every candidate so far."""
    beams: dict[tuple[str, ...], tuple[float, tuple[str, ...]]] = {(): (0.0, ())}
    for ev in evidence:
        options = scores(ev, deck, levels)
        grown: dict[tuple[str, ...], tuple[float, tuple[str, ...]]] = {}
        for recent, (total, path) in beams.items():
            for name, s in options.items():
                step = total + s - (REPEAT_COST if name in recent else 0.0)
                # no play leaves the cycle where it was
                key = recent if name == NONE else (recent + (name,))[-4:]
                if key not in grown or step > grown[key][0]:
                    grown[key] = (step, path + (name,))
        beams = dict(sorted(grown.items(), key=lambda kv: kv[1][0], reverse=True)[:BEAM])
    return list(max(beams.values(), key=lambda v: v[0])[1])
