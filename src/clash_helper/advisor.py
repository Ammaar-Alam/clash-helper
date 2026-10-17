"""Pick the single most useful call for this moment."""

from __future__ import annotations

from dataclasses import dataclass

from clash_helper.cards import Card, answers, card
from clash_helper.tracker import Opponent

SCREEN_MID_X = 332
THREAT_SECONDS = 7.0
LOW_ELIXIR = 3.5


@dataclass(frozen=True)
class Call:
    action: str
    reason: str = ""
    urgent: bool = False

    def __str__(self) -> str:
        return f"{self.action}: {self.reason}" if self.reason else self.action


@dataclass
class Threat:
    card: str
    x: int
    time: float

    @property
    def lane(self) -> str:
        return "left" if self.x < SCREEN_MID_X else "right"


def _other(lane: str) -> str:
    return "right" if lane == "left" else "left"


def _defend_spot(threat: Card, answer: Card, lane: str) -> str:
    if answer.spell:
        return f"on the {threat.name}"
    if answer.building:
        return "center, pull it in"
    if threat.has("win") and not threat.has("air"):
        return f"center, {lane} side"
    return f"in front of {lane} tower"


def advise(
    now: float,
    hand: list[Card | None],
    elixir: int,
    opp: Opponent,
    threat: Threat | None,
    answered: bool,
) -> Call:
    known = [c for c in hand if c is not None]
    their_hand = [card(n) for n in opp.hand]
    their_elixir = int(opp.elixir)

    # 1 an enemy push nobody has answered yet
    enemy = card(threat.card) if threat else None
    # a cheap unit that does not go for towers is the tower's job, spending on it loses elixir
    worth_answering = enemy is not None and (enemy.cost > 2 or enemy.has("win") or enemy.has("swarm"))
    if threat and worth_answering and now - threat.time <= THREAT_SECONDS and not answered:
        options = sorted((c for c in known if answers(enemy, c)), key=lambda c: c.cost)
        affordable = [c for c in options if c.cost <= elixir]
        if affordable:
            pick = affordable[0]
            return Call(f"{pick.name}, {_defend_spot(enemy, pick, threat.lane)}", f"stop their {enemy.name}", urgent=True)
        if options:
            return Call(f"Save for {options[0].name}", f"{enemy.name} coming {threat.lane}", urgent=True)
        if not known:
            return Call(f"Defend {threat.lane}", f"{enemy.name} coming", urgent=True)

    attackers = [c for c in known if c.has("win") and c.cost <= elixir]
    push_lane = _other(threat.lane) if threat else "right"

    # 2 they just spent big and cannot answer
    if their_elixir <= LOW_ELIXIR and elixir >= 4:
        if attackers:
            pick = max(attackers, key=lambda c: c.cost)
            return Call(f"{pick.name}, {push_lane} bridge", f"they're at {their_elixir} elixir")
        if not known:
            return Call(f"Attack {push_lane}", f"they're at {their_elixir} elixir")

    # 3 their answer to our attacker is out of hand
    if opp.deck_complete and attackers:
        for pick in sorted(attackers, key=lambda c: -c.cost):
            counters = [c for c in their_hand if answers(pick, c) and c.cost <= their_elixir]
            if not counters:
                return Call(f"{pick.name}, {push_lane} bridge", "no answer in their hand")

    # 4 do not waste elixir at the cap
    if elixir >= 9:
        tanks = [c for c in known if c.has("tank") and c.cost <= elixir]
        if tanks:
            return Call(f"{tanks[0].name}, back {push_lane}", "you're full, start a push")
        cheap = sorted((c for c in known if not c.spell), key=lambda c: c.cost)
        if cheap:
            return Call(f"{cheap[0].name}, back {push_lane}", "you're full")
        return Call("Play something", "you're full")

    # 5 nothing to do, say why waiting is right
    held = [c for c in their_hand if c.spell and c.cost <= their_elixir]
    if held and opp.deck_complete:
        return Call("Wait", f"{held[0].name} in their hand")
    return Call("Wait", f"they have {their_elixir} elixir")
