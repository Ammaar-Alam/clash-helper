"""Pick the single most useful call for this moment."""

from __future__ import annotations

from dataclasses import dataclass

from clash_helper import playbook
from clash_helper.board import Board
from clash_helper.cards import Card, answers, card
from clash_helper.tracker import Opponent

SCREEN_MID_X = 332
THREAT_SECONDS = 7.0
LOW_ELIXIR = 3.5
LOW_TOWER = 800
SUPPORT_SECONDS = (2.0, 9.0)  # after a tank lands, the window to put support behind it
# ponytail: rough crown tower damage at level 11, replace with per level stats if finishing calls misfire
TOWER_DAMAGE = {"Rocket": 450, "Lightning": 250, "Fireball": 200, "Poison": 150, "Arrows": 100, "Earthquake": 100,
                "The Log": 70, "Zap": 60, "Giant Snowball": 50, "Barbarian Barrel": 60}


@dataclass(frozen=True)
class Call:
    action: str
    reason: str = ""
    urgent: bool = False
    rule: str = "wait"  # which rule made the call, used to measure what works

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


@dataclass
class MyPlay:
    card: str
    time: float
    lane: str | None = None


def _other(lane: str) -> str:
    return "right" if lane == "left" else "left"


def _defend_spot(threat: Card, answer: Card, lane: str) -> str:
    if answer.spell:
        return f"on the {threat.name}"
    if answer.building:
        return playbook.spot("pull", lane, threat.name)
    if threat.has("win") and not threat.has("air"):
        return f"center, {lane} side"
    return playbook.spot("front", lane, threat.name)


def _defense(enemy: Card, lane: str, known: list[Card], elixir: int) -> Call | None:
    """Best answer in hand, the playbook first and rough roles second."""
    names = {c.name: c for c in known}
    picks = [(names[n], playbook.spot(kind, lane, enemy.name)) for n, kind in playbook.answers_for(enemy.name) if n in names]
    picks += [(c, _defend_spot(enemy, c, lane)) for c in sorted(known, key=lambda c: c.cost)
              if answers(enemy, c) and c.name not in {p.name for p, _ in picks}]
    affordable = [(c, where) for c, where in picks if c.cost <= elixir]
    if affordable:
        pick, where = affordable[0]
        return Call(f"{pick.name}, {where}", f"stop their {enemy.name}", urgent=True, rule="defend")
    if picks:
        return Call(f"Save for {picks[0][0].name}", f"{enemy.name} coming {lane}", urgent=True, rule="save")
    return None


def _push_lane(board: Board | None, threat: Threat | None) -> str:
    """Attack the weaker tower, otherwise away from their last push."""
    if board:
        hp = {ln: board.their_towers[ln] for ln in ("left", "right") if board.their_towers[ln]}
        if len(hp) == 2 and hp["left"] != hp["right"]:
            return min(hp, key=hp.get)
        if len(hp) == 1:
            return next(iter(hp))
    return _other(threat.lane) if threat else "right"


def advise(
    now: float,
    hand: list[Card | None],
    elixir: int,
    opp: Opponent,
    threat: Threat | None,
    answered: bool,
    board: Board | None = None,
    mine: MyPlay | None = None,
) -> Call:
    known = [c for c in hand if c is not None]
    their_hand = [card(n) for n in opp.hand]
    their_elixir = int(opp.elixir)

    # 1 an enemy push nobody has answered yet
    enemy = card(threat.card) if threat else None
    # a cheap unit that does not go for towers is the tower's job, spending on it loses elixir
    worth_answering = enemy is not None and (enemy.cost > 2 or enemy.has("win") or enemy.has("swarm"))
    if threat and worth_answering and now - threat.time <= THREAT_SECONDS and not answered:
        call = _defense(enemy, threat.lane, known, elixir)
        if call:
            return call
        if not known:
            return Call(f"Defend {threat.lane}", f"{enemy.name} coming", urgent=True, rule="defend")

    # 2 enemy units already on your side, maybe ones we never saw land
    if board and not answered:
        for ln in sorted(board.enemy_home, key=lambda ln: -board.enemy_home[ln]):
            if board.enemy_home[ln] >= 2 or (board.enemy_home[ln] and (board.my_towers[ln] or 9999) < LOW_TOWER):
                low = board.my_towers[ln] is not None and board.my_towers[ln] < LOW_TOWER
                return Call(f"Defend {ln}", f"your tower is at {board.my_towers[ln]}" if low else "they crossed the river",
                            urgent=True, rule="defend")

    lane = _push_lane(board, threat)

    # 3 your tank is walking, put support behind it while it soaks damage
    if mine and SUPPORT_SECONDS[0] <= now - mine.time <= SUPPORT_SECONDS[1] and card(mine.card).has("tank"):
        support = [c for c in known if not c.spell and not c.building and not c.has("tank") and c.cost <= elixir]
        if support:
            pick = max(support, key=lambda c: (c.has("splash"), c.has("aa"), c.cost))
            side = mine.lane or lane
            return Call(f"{pick.name}, behind your {mine.card}", f"support the {side} push", rule="support")

    attackers = [c for c in known if c.has("win") and c.cost <= elixir]

    # 4 they just spent big and cannot answer
    if their_elixir <= LOW_ELIXIR and elixir >= 4:
        if attackers:
            pick = max(attackers, key=lambda c: c.cost)
            where = "bridge" if not pick.has("tank") else "bridge, then support it"
            return Call(f"{pick.name}, {lane} {where}", f"they're at {their_elixir} elixir", rule="punish")
        if not known:
            return Call(f"Attack {lane}", f"they're at {their_elixir} elixir", rule="punish")

    # 5 a damaged tower one spell from falling
    if board:
        for ln, hp in board.their_towers.items():
            spells = [c for c in known if c.spell and c.cost <= elixir and hp and TOWER_DAMAGE.get(c.name, 0) >= hp]
            if spells:
                pick = min(spells, key=lambda c: c.cost)
                return Call(f"{pick.name} their {ln} tower", f"only {hp} left", rule="finish")

    # 6 their answer to our attacker is out of hand
    if opp.deck_complete and attackers:
        for pick in sorted(attackers, key=lambda c: -c.cost):
            counters = [c for c in their_hand if answers(pick, c) and c.cost <= their_elixir]
            if not counters:
                return Call(f"{pick.name}, {lane} bridge", "no answer in their hand", rule="safe")

    # 7 do not waste elixir at the cap
    if elixir >= 9:
        tanks = [c for c in known if c.has("tank") and c.cost <= elixir]
        if tanks:
            return Call(f"{tanks[0].name}, back {lane}", "you're full, start a push", rule="full")
        cheap = sorted((c for c in known if not c.spell), key=lambda c: c.cost)
        if cheap:
            return Call(f"{cheap[0].name}, back {lane}", "you're full", rule="full")
        return Call("Play something", "you're full", rule="full")

    # 8 nothing to do, say why waiting is right
    held = [c for c in their_hand if c.spell and c.cost <= their_elixir]
    if held and opp.deck_complete:
        return Call("Wait", f"{held[0].name} in their hand")
    if their_elixir >= 7:
        return Call("Wait", "they can answer anything now")
    return Call("Wait", "build up elixir")
