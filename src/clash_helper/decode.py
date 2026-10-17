"""Name a whole sequence of enemy plays at once using the card cycle.

A card can only be played from the hand, and the hand is the deck minus the
last 4 plays. So among any 4 plays in a row no card repeats. Each play has
weak evidence on its own, the cycle ties them together, and a beam search
finds the most likely sequence. Breaking the rule is allowed at a cost
because spells and missed plays are invisible.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from clash_helper.cards import card
from clash_helper.identify import unit_cards

BEAM = 300
REPEAT_COST = 3.0  # nats, a repeat within 4 plays needs strong evidence
NAMED_WEIGHT = 3.0


@dataclass
class Evidence:
    votes: Counter  # detector unit -> summed confidence
    units: int  # level badges around the landing spot
    digits: int | None  # digits in the level badge
    named: str | None = None  # the vision model's answer, arrives a moment later


def scores(evidence: Evidence, deck: list[str], levels: dict[str, int]) -> dict[str, float]:
    """Log score per deck card for one play, 0 means no opinion."""
    per_card: Counter = Counter()
    for unit, weight in evidence.votes.items():
        for name in unit_cards().get(unit, []):
            if name in deck:
                per_card[name] += weight * (1.0 if card(name).units[0] == unit else 0.6)
    total = sum(per_card.values())
    out = {}
    for name in deck:
        s = math.log((per_card[name] + 0.1) / (total + 0.8)) if total else 0.0
        count = card(name).count
        if evidence.units >= 2:
            s += 1.0 if count == evidence.units else -1.0
        elif evidence.units == 1 and count >= 2:
            s -= 0.4
        if evidence.digits and name in levels:
            s += 0.5 if len(str(levels[name])) == evidence.digits else -0.5
        if evidence.named == name:
            # strong but not final, the cycle can still overrule a slip
            s += NAMED_WEIGHT
        if card(name).spell:
            s -= 5.0  # spells never show a deploy timer
        out[name] = s
    return out


def decode(evidence: list[Evidence], deck: list[str], levels: dict[str, int]) -> list[str]:
    """Most likely card for every play so far."""
    beams: dict[tuple[str, ...], tuple[float, tuple[str, ...]]] = {(): (0.0, ())}
    for ev in evidence:
        options = scores(ev, deck, levels)
        grown: dict[tuple[str, ...], tuple[float, tuple[str, ...]]] = {}
        for recent, (total, path) in beams.items():
            for name, s in options.items():
                step = total + s - (REPEAT_COST if name in recent else 0.0)
                key = (recent + (name,))[-4:]
                if key not in grown or step > grown[key][0]:
                    grown[key] = (step, path + (name,))
        best = sorted(grown.items(), key=lambda kv: kv[1][0], reverse=True)[:BEAM]
        beams = dict(best)
    return list(max(beams.values(), key=lambda v: v[0])[1])
