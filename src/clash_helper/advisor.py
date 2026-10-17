"""Pick the single most useful instruction for this moment, for any deck.

Rules run in priority order. Every call names a card and an exact spot when it
asks for a play, and every call carries the plan for what comes next so the
player always knows what they are waiting for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from clash_helper import playbook
from clash_helper.board import LANE_COL, Board, lane, row_of, tile
from clash_helper.vision import RIVER_Y
from clash_helper.cards import Card, answers, card
from clash_helper.tracker import Opponent

SCREEN_MID_X = 332
SECONDS_PER_ELIXIR = 2.8
THREAT_SECONDS = 20.0  # tracking drops a threat once it dies, this only caps a stuck one
LOW_ELIXIR = 3.5
LOW_TOWER = 800
SUPPORT_SECONDS = (1.5, 9.0)  # after a tank lands, the window to put support behind it
CLUSTER_RADIUS = 80  # px, enemy units this close share one spell
# ponytail: rough crown tower damage at level 11, replace with per level stats if finishing calls misfire
TOWER_DAMAGE = {"Rocket": 450, "Lightning": 250, "Fireball": 200, "Poison": 150, "Arrows": 100, "Earthquake": 100,
                "The Log": 70, "Zap": 60, "Giant Snowball": 50, "Barbarian Barrel": 60}
ROLLING = {"The Log", "Barbarian Barrel"}
SWARM_SPELLS = {"The Log", "Arrows", "Zap", "Giant Snowball", "Barbarian Barrel", "Fireball", "Poison", "Tornado"}


@dataclass(frozen=True)
class Call:
    action: str
    reason: str = ""
    urgent: bool = False
    rule: str = "wait"  # which rule made the call, used to measure what works
    card: str | None = None  # the card to play, None when nothing should be played
    target: tuple[int, int] | None = None  # where to drop it on the canonical screen
    plan: str = ""  # what comes after this

    def __str__(self) -> str:
        return f"{self.action}: {self.reason}" if self.reason else self.action


@dataclass
class Threat:
    card: str
    x: int
    time: float
    y: int = 400
    last_seen: float = 0.0  # tracked every frame, gone once it stops being seen
    track: int | None = None  # the detector's id for the unit, once it is known

    @property
    def lane(self) -> str:
        return "left" if self.x < SCREEN_MID_X else "right"


@dataclass
class MyPlay:
    card: str
    time: float
    lane: str | None = None
    at: tuple[int, int] | None = None


@dataclass
class Situation:
    now: float
    hand: list[Card | None]
    elixir: int
    opp: Opponent
    threat: Threat | None = None
    answered: bool = True
    board: Board | None = None
    mine: MyPlay | None = None
    multiplier: float = 1.0
    enemies: list[tuple[int, int]] = field(default_factory=list)  # enemy unit positions right now
    last_rule: str = ""  # the rule behind the previous call, for calls that should not flicker

    @property
    def known(self) -> list[Card]:
        return [c for c in self.hand if c is not None]

    def affordable(self, cards: list[Card]) -> list[Card]:
        return [c for c in cards if c.cost <= self.elixir]

    def seconds_until(self, cost: int) -> float:
        """How long until the player can pay for a card."""
        return max(0.0, (cost - self.elixir) * SECONDS_PER_ELIXIR / self.multiplier)


def _other(side: str) -> str:
    return "right" if side == "left" else "left"


def _play(c: Card, where_text: str, where: tuple[int, int], reason: str, rule: str,
          urgent: bool = False, plan: str = "") -> Call:
    return Call(f"{c.name}, {where_text}", reason, urgent, rule, c.name, where, plan)


def _threat_spot(s: Situation) -> tuple[int, int] | None:
    """Where the threat is now, the closest enemy unit to where it landed."""
    if not s.threat:
        return None
    near = [(x, y) for x, y in s.enemies if lane(x) == s.threat.lane]
    if not near:
        return (s.threat.x, s.threat.y)
    return min(near, key=lambda p: math.hypot(p[0] - s.threat.x, p[1] - s.threat.y) - 0.5 * p[1])


def _valid(kind: str, row: float) -> bool:
    """Whether a spot still makes sense with the threat at this row, the river is row 15."""
    return {
        "pull": row >= 12,  # a building only pulls what has not reached it yet
        "bridge": row >= 14,
        "front": row >= 11,
    }.get(kind, True)


def _defense(s: Situation, enemy: Card, side: str, at: tuple[int, int] | None) -> Call | None:
    """Best answer in hand for where the threat is right now, the playbook first and rough roles second."""
    row = row_of(at[1]) if at else 20.0
    names = {c.name: c for c in s.known}
    picks = [(names[n], kind) for n, kind in playbook.answers_for(enemy.name) if n in names]
    taken = {c.name for c, _ in picks}
    for c in sorted(s.known, key=lambda c: c.cost):
        if c.name not in taken and answers(enemy, c):
            kind = "on" if c.spell else "pull" if c.building else "front"
            picks.append((c, kind))
    # once it is past a spot, troops go straight onto it instead and buildings are too late
    picks = [(c, kind if _valid(kind, row) else "on") for c, kind in picks if _valid(kind, row) or not c.building]
    ready = [(c, kind) for c, kind in picks if c.cost <= s.elixir]
    if ready:
        pick, kind = ready[0]
        where = playbook.spot_xy(kind, side, at)
        if kind == "on" and not pick.spell and at:
            # a troop lands just on your side of it so it fights instead of chasing
            where = (at[0], min(1092, at[1] + 30))
        if pick.name in ROLLING and kind == "on":
            # it rolls up the lane, so it starts a little on your side of the target
            where = (where[0], where[1] + 50)
        text = playbook.spot(kind, side, enemy.name) if kind != "on" or pick.spell else f"right on the {enemy.name}"
        return _play(pick, text, where, f"stop their {enemy.name}", "defend", urgent=True)
    if picks:
        pick, kind = picks[0]
        wait = s.seconds_until(pick.cost)
        return Call(f"Get {pick.name} ready", f"{enemy.name} coming {side}", True, "save", None, None,
                    f"{pick.name} {playbook.spot(kind, side, enemy.name)} in {wait:.0f}s")
    return None


def _push_lane(s: Situation) -> str:
    """Attack the weaker tower, otherwise away from their last push."""
    if s.board:
        hp = {ln: s.board.their_towers[ln] for ln in ("left", "right") if s.board.their_towers[ln]}
        if len(hp) == 2 and hp["left"] != hp["right"]:
            return min(hp, key=hp.get)
        if len(hp) == 1:
            return next(iter(hp))
    return _other(s.threat.lane) if s.threat else "right"


def _cluster(s: Situation) -> tuple[tuple[int, int], int] | None:
    """The densest group of enemy units and its size."""
    best = None
    # only groups pushing into your half are worth a spell, the rest are their problem
    near = [(x, y) for x, y in s.enemies if y > RIVER_Y - 60]
    for x, y in near:
        group = [(a, b) for a, b in near if math.hypot(a - x, b - y) <= CLUSTER_RADIUS]
        if best is None or len(group) > best[1]:
            cx = round(sum(a for a, _ in group) / len(group))
            cy = round(sum(b for _, b in group) / len(group))
            best = ((cx, cy), len(group))
    return best


def _attackers(s: Situation) -> list[Card]:
    return [c for c in s.known if c.has("win")]


def _plan(s: Situation, side: str) -> str:
    """The next thing worth saving for, so waiting always has a point."""
    wins = sorted(_attackers(s), key=lambda c: -c.cost)
    if wins:
        pick = wins[0]
        where = "behind your king tower" if pick.has("tank") and s.multiplier == 1 else f"{side} bridge"
        wait = s.seconds_until(pick.cost + (2 if pick.has("tank") else 0))
        return f"{pick.name} {where} in {wait:.0f}s" if wait > 0 else f"{pick.name} {where} now"
    cycle = sorted((c for c in s.known if not c.spell and not c.building), key=lambda c: c.cost)
    if cycle:
        return f"cycle {cycle[0].name} at the back to find your win card"
    return ""


def advise(s: Situation) -> Call:
    known = s.known
    their_hand = [card(n) for n in s.opp.hand]
    their_elixir = int(s.opp.elixir)
    side = _push_lane(s)
    plan = _plan(s, side)

    # 1 an enemy push nobody has answered yet
    enemy = card(s.threat.card) if s.threat else None
    # a cheap unit that does not go for towers is the tower's job, spending on it loses elixir
    worth = enemy is not None and (enemy.cost > 2 or enemy.has("win") or enemy.has("swarm"))
    # a tank still walking up their side is not a defense problem yet
    close = s.threat is not None and row_of(s.threat.y) <= 18
    if s.threat and worth and close and s.now - s.threat.time <= THREAT_SECONDS and not s.answered:
        call = _defense(s, enemy, lane(s.threat.x), (s.threat.x, s.threat.y))
        if call:
            return call
        if not known:
            return Call(f"Defend {s.threat.lane}", f"{enemy.name} coming", True, "defend")

    # 2 enemy units already on your side, maybe ones we never saw land
    if s.board and not s.answered:
        for ln in sorted(s.board.enemy_home, key=lambda ln: -s.board.enemy_home[ln]):
            hp = s.board.my_towers[ln]
            low = hp is not None and 0 < hp < LOW_TOWER
            if s.board.enemy_home[ln] >= 2 or (s.board.enemy_home[ln] and (low or hp == 0)):
                # aim at the units themselves, a fixed spot is useless once they are deep in
                crossed = [(x, y) for x, y in s.enemies if lane(x) == ln and y > RIVER_Y]
                at = max(crossed, key=lambda p: p[1]) if crossed else playbook.spot_xy("front", ln)
                deep = row_of(at[1]) < 12
                fighters = [c for c in known if not c.spell and (c.has("splash") or c.has("dps") or (c.building and not deep))]
                defenders = sorted(s.affordable(fighters), key=lambda c: (not c.has("splash"), c.cost))
                reason = ("they're on your king tower" if hp == 0 else f"your tower is at {hp}" if low
                          else "they crossed the river")
                if defenders:
                    c = defenders[0]
                    where = playbook.spot_xy("pull", ln) if c.building else (at[0], min(1092, at[1] + 30))
                    text = playbook.spot("pull", ln, "") if c.building else f"right on them, {ln} side"
                    return _play(c, text, where, reason, "defend", urgent=True)
                return Call(f"Defend {ln}", reason, True, "defend", plan=plan)

    # 3 several enemy units bunched up, one spell takes them all
    group = _cluster(s)
    # a group needs 3 units to call for a spell, and keeps the call while 2 remain so it does not flicker
    if group and group[1] >= (2 if s.last_rule == "spell" else 3):
        spells = sorted(s.affordable([c for c in known if c.name in SWARM_SPELLS]), key=lambda c: c.cost)
        if spells:
            return _play(spells[0], "on their group", group[0], f"{group[1]} units in one spot", "spell")

    # 4 your tank is walking, put support behind it while it soaks damage
    if s.mine and SUPPORT_SECONDS[0] <= s.now - s.mine.time <= SUPPORT_SECONDS[1] and card(s.mine.card).has("tank"):
        support = s.affordable([c for c in known if not c.spell and not c.building and not c.has("tank")])
        if support:
            pick = max(support, key=lambda c: (c.has("splash"), c.has("aa"), c.cost))
            push = s.mine.lane or side
            behind = s.mine.at or tile(LANE_COL[push], 9)
            spot = (behind[0], min(1092, behind[1] + 60))
            return _play(pick, f"behind your {s.mine.card}", spot, f"support the {push} push", "support",
                         plan="then hold elixir for their answer")

    attackers = s.affordable(_attackers(s))

    # 5 you just defended and your survivors are still alive, send the attacker with them
    if s.board and attackers:
        for ln in ("left", "right"):
            if s.board.mine_home[ln] >= 1 and not s.board.enemy_home[ln] and s.answered and s.threat and s.threat.lane == ln:
                pick = max(attackers, key=lambda c: c.cost)
                return _play(pick, f"{ln} bridge", playbook.spot_xy("bridge", ln), "counter-push with your survivors", "counter")

    # 6 they just spent big and cannot answer
    if their_elixir <= LOW_ELIXIR and s.elixir >= 4:
        if attackers:
            pick = max(attackers, key=lambda c: c.cost)
            return _play(pick, f"{side} bridge", playbook.spot_xy("bridge", side), f"they're at {their_elixir} elixir", "punish",
                         plan="support it next" if pick.has("tank") else "")
        if not known:
            return Call(f"Attack {side}", f"they're at {their_elixir} elixir", rule="punish")

    # 7 a damaged tower one spell from falling
    if s.board:
        for ln, hp in s.board.their_towers.items():
            spells = [c for c in s.affordable(known) if c.spell and hp and TOWER_DAMAGE.get(c.name, 0) >= hp]
            if spells:
                pick = min(spells, key=lambda c: c.cost)
                tower = tile(LANE_COL[ln], 25.5)
                return _play(pick, f"their {ln} tower", tower, f"only {hp} left", "finish")

    # 8 their answer to your attacker is out of their hand
    if s.opp.deck_complete and attackers:
        # a lone tank at the bridge wastes it, tanks start from the back in rule 9
        for pick in sorted((c for c in attackers if not c.has("tank")), key=lambda c: -c.cost):
            if not [c for c in their_hand if answers(pick, c) and c.cost <= their_elixir]:
                return _play(pick, f"{side} bridge", playbook.spot_xy("bridge", side), "no answer in their hand", "safe")

    # 9 build a big push from the back while there is time to walk it up
    tanks = [c for c in known if c.has("tank") and c.has("win")]
    if tanks:
        tank = tanks[0]
        enough = tank.cost + (1 if s.multiplier > 1 else 3)
        if s.elixir >= min(10, enough):
            return _play(tank, f"behind your king tower, {side} side", playbook.spot_xy("back", side),
                         "start a push while it walks", "push", plan="support it at the bridge")

    # 10 your attacker is not in hand, cycle a cheap card to bring it back
    if not _attackers(s) and s.elixir >= 7:
        cheap = sorted((c for c in known if not c.spell and not c.building), key=lambda c: c.cost)
        if cheap:
            return _play(cheap[0], f"behind your {side} tower", playbook.spot_xy("behind", side),
                         "cycle toward your win card", "cycle")

    # 11 do not waste elixir at the cap
    if s.elixir >= 9:
        cheap = sorted((c for c in known if not c.spell), key=lambda c: c.cost)
        if cheap:
            return _play(cheap[0], f"behind your {side} tower", playbook.spot_xy("behind", side), "you're full", "full")
        return Call("Play something", "you're full", rule="full")

    # 12 nothing to play yet, say what you are saving for and what to watch
    held = [c for c in their_hand if c.spell and c.cost <= their_elixir]
    reason = f"they hold {held[0].name}" if held and s.opp.deck_complete else (
        "they can answer anything now" if their_elixir >= 7 else "build up elixir")
    return Call("Wait", reason, rule="wait", plan=plan)
