"""One match from frames in to calls out, shared by live play and replays.

The KataCR detector sees every unit and spell with its side and a track id.
The red deploy timer says exactly when and where an enemy card lands, the
tracked units around it say what it was and follow it down the lane.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from clash_helper import api, kcr, vision
from clash_helper.advisor import Call, MyPlay, Situation, Threat, advise
from clash_helper.board import LANES, Board, BoardReader, lane
from clash_helper.cards import card
from clash_helper.decode import NONE, Evidence, best, decode
from clash_helper.hand import HandReader
from clash_helper.tracker import UNKNOWN, Opponent

NAME_SECONDS = 1.4  # how long after a deploy timer the units around it vote on the card
CLOCK_MERGE = 70  # px, one deploy timer seen over several frames
CLOCK_SECONDS = 1.6
MATCH_GONE_SECONDS = 3.0
THREAT_LOST = 1.2  # seconds a tracked unit can go unseen before it counts as dead
SAME_SPELL_SECONDS = 2.0
SPELL_SIGHTINGS = 2  # enemy side detections of a spell before it counts as cast
SURE_SPELL = 0.75  # or one detection this confident
BIRTH_CONF = 0.6  # a new enemy unit this confident is a play even without a visible timer
REAPPEAR = (4.0, 220)  # seconds and px within which a "new" unit is an old one the tracker lost
# cards that put the same units on the arena as another card, a child near its parent is no new play
SPAWNS = {
    "Skeletons": {"Witch", "Tombstone", "Graveyard", "Skeleton King", "Skeleton Army", "Skeleton Barrel", "Giant Skeleton"},
    "Goblins": {"Goblin Hut", "Goblin Giant", "Goblin Barrel", "Goblin Gang", "Goblin Machine", "Goblin Demolisher", "Goblinstein"},
    "Spear Goblins": {"Goblin Hut", "Goblin Giant", "Goblin Gang"},
    "Barbarians": {"Barbarian Hut", "Barbarian Barrel", "Battle Ram"},
    "Bats": {"Night Witch"},
    "Golem": {"Golem"},
    "Lava Hound": {"Lava Hound"},
    "Elixir Golem": {"Elixir Golem"},
    "Phoenix": {"Phoenix"},
}
SPAWN_RADIUS = 260
# cards whose units drop in with a deploy timer of their own when the card bursts or dies
DEATH_SPAWNERS = {"Skeleton Barrel", "Battle Ram", "Golem", "Elixir Golem", "Lava Hound", "Phoenix", "Goblin Cage",
                  "Giant Skeleton", "Goblin Giant", "Goblin Barrel", "Barbarian Barrel", "Royal Delivery", "Graveyard"}
# spells that put units on the arena, their units land wherever the spell does
SPELL_SPAWNS = {"Goblin Barrel": {"Goblins"}, "Graveyard": {"Skeletons"}, "Barbarian Barrel": {"Barbarians"},
                "Royal Delivery": {"Royal Recruits"}, "Skeleton Barrel": {"Skeletons"}}
SPELL_SPAWN_SECONDS = 4.0
POCKET_Y = 790  # how far into your side they may deploy once your tower in that lane is down
VOTE_AGE = 0.5  # only units that appeared around the landing vote, not bystanders already there
ANYWHERE = {"Miner", "Goblin Drill", "Royal Delivery"}  # land on your side without a destroyed tower
SURE_VOTES = 1.2  # summed confidence before a name may join a deck learned from play
LOOKUP_EVERY = 1.0  # seconds between name reads while the intro fades
LOOKUP_WINDOW = 25.0


@dataclass
class Pending:
    x: int
    y: int
    time: float
    multiplier: float
    votes: Counter = field(default_factory=Counter)  # card -> summed confidence
    tracks: set[int] = field(default_factory=set)
    source: str = "timer"  # timer, birth or spell


@dataclass
class Track:
    side: int = 0  # votes, above zero is the opponent, single frames flip now and then
    cards: Counter = field(default_factory=Counter)
    x: int = 0
    y: int = 0
    first: float = 0.0
    last: float = 0.0
    born_enemy: bool = False  # side on its first sighting, before votes settle
    sightings: int = 0

    @property
    def enemy(self) -> bool:
        return self.side > 0

    @property
    def card(self) -> str | None:
        return self.cards.most_common(1)[0][0] if self.cards else None


@dataclass
class View:
    """Everything the panel shows."""

    in_match: bool = False
    call: Call = field(default_factory=lambda: Call("Waiting for a match"))
    their_hand: list[str] = field(default_factory=list)
    maybe: list[str] = field(default_factory=list)
    next_card: str | None = None
    their_elixir: float = 0
    my_elixir: int = 0
    deck_known: bool = False
    plays: list[str] = field(default_factory=list)
    queue: list[str | None] = field(default_factory=list)
    opponent: str = ""
    board: Board | None = None
    my_hand: list[str | None] = field(default_factory=list)


class Match:
    def __init__(self, detector: kcr.Detector | None = None, lookup: bool = True, sync: bool = False) -> None:
        """`sync` waits for the detector on every frame, for replays that must not skip any."""
        if detector is None and not kcr.ready():
            raise SystemExit("The unit detector is not installed yet, run: clash-helper setup")
        self.detector = detector or kcr.Detector()
        self.lookup = lookup
        self.sync = sync
        self.hand_reader: HandReader | None = None
        self.my_deck: list[dict] = []
        self.ended: dict = {}
        self._reset()

    def _load_my_deck(self) -> None:
        """Read the deck being played right now, so switching decks between matches just works."""
        tag = api.setting("CR_PLAYER_TAG")
        if not (self.lookup and tag and api.token()):
            return
        try:
            deck = api.player(tag)["currentDeck"]
        except (api.ApiError, OSError) as error:
            print(f"could not load your deck: {error}", flush=True)
            return
        if [c["name"] for c in deck] != [c["name"] for c in self.my_deck]:
            self.hand_reader = HandReader(deck)
            self.my_deck = deck

    def _reset(self) -> None:
        self.opp = Opponent()
        self.deck_from_api = False
        self.started: float | None = None
        self.last_seen = 0.0
        self.pending: list[Pending] = []
        self.clocks: list[tuple[int, int, float]] = []
        self.threat: Threat | None = None
        self.answered = True
        self.my_elixir = 5
        self.lookup_state = "todo"  # todo, running, done
        self.levels: dict[str, int] = {}
        self.opponent = ""
        self.lookup_note = ""
        self.last_ocr = -10.0
        self.my_hand: list = [None] * 4
        self.played = 0
        self.events: list[tuple[Pending, Evidence]] = []
        self.debug: list = []
        self.last_play = ""
        self.reader = BoardReader()
        self.mine: MyPlay | None = None
        self.my_plays: list[MyPlay] = []
        self.last_drop = -10.0
        self.multiplier = 1.0
        self.last_rule = ""
        self.enemies: list[tuple[int, int]] = []
        self.tracks: dict[int, Track] = {}
        self.spells_seen: dict[int, float] = {}
        self.spell_casts: list[tuple[str, float]] = []
        self.last_seen_units = 0.0

    def _lookup_opponent(self, frame: np.ndarray, now: float) -> None:
        """Read their name and clan, then fetch the deck they are using right now."""
        if now - self.last_ocr < LOOKUP_EVERY or now - (self.started or now) > LOOKUP_WINDOW:
            return
        self.last_ocr = now
        # name, clan, trophies, the clan line says No Clan for clanless players
        read = vision.read_text(frame[95:200, 55:320])
        lines = [t for t in read if not t.replace(" ", "").isdigit()]
        trophies = [int(t.replace(" ", "")) for t in read if t.replace(" ", "").isdigit()]
        if len(lines) < 2 or not trophies:
            return
        name, clan = lines[0], lines[1]
        self.opponent = name
        self.lookup_note = f"read {name!r} in clan {clan!r} with {trophies[0]} trophies"
        if clan.lower() == "no clan":
            # the API has no name search, a clanless player cannot be found
            self.lookup_state = "done"
            return
        self.lookup_state = "running"

        def run() -> None:
            try:
                found = api.find_player(name, clan)
            except api.ApiError as error:
                print(f"opponent lookup failed: {error}", flush=True)
                found = None
            # a namesake in a same named clan would load a stranger's deck, the trophies must agree too
            if found and abs(found.get("trophies", 0) - trophies[0]) > 60:
                self.lookup_note += f", found {found['name']} with {found.get('trophies')} trophies so ignored them"
                found = None
            elif not found:
                self.lookup_note += ", no unique match in that clan"
            if found and found.get("currentDeck"):
                self.opp.preload([c["name"] for c in found["currentDeck"]])
                self.deck_from_api = True
                # the badge shows the level counted from 16 down for every rarity
                self.levels = {c["name"]: c["level"] + 16 - c["maxLevel"] for c in found["currentDeck"]}
                self.lookup_note += f", deck {', '.join(self.opp.deck)}"
            self.lookup_state = "done"

        threading.Thread(target=run, daemon=True).start()

    def step(self, frame: np.ndarray, now: float) -> View:
        if not vision.in_match(frame):
            if self.started is not None and now - self.last_seen > MATCH_GONE_SECONDS:
                # keep what happened for the journal, the next match starts clean
                self.ended = {"plays": list(self.debug), "opponent": self.opponent, "deck": list(self.opp.deck),
                              "lookup": self.lookup_note, "deck_from_api": self.deck_from_api}
                self._reset()
            return View(in_match=False) if self.started is None else self._view(now)
        if self.started is None:
            self.started = now
            self.opp.last_time = now
            threading.Thread(target=self._load_my_deck, daemon=True).start()
        self.last_seen = now
        if self.lookup and self.lookup_state == "todo" and api.token():
            self._lookup_opponent(frame, now)

        multiplier = self.multiplier = vision.elixir_multiplier(frame)
        self.opp.tick(now, multiplier)
        self._track_my_elixir(vision.own_elixir(frame), now)
        if self.hand_reader:
            hand, _ = self.hand_reader.read(frame)
            # a card mid-slide or being dragged reads as None, keep the last good name
            hand = [new or old for new, old in zip(hand, self.my_hand)]
            self._track_my_plays(hand, now)
            self.my_hand = hand

        # the detector works in the background, every new result updates the board
        self.detector.submit(frame, now)
        while self.sync and self.detector.busy:
            time.sleep(0.002)
        seen_at, units = self.detector.result()
        if seen_at > self.last_seen_units:
            self.last_seen_units = seen_at
            self._see(units, seen_at)
        self.reader.towers(frame, now)

        for x, y in vision.deploy_clocks(frame):
            if y > vision.ARENA_BOTTOM - 120 or x > 615:
                continue  # emote bubbles and the crown counter look like timers
            if any(abs(x - cx) + abs(y - cy) < CLOCK_MERGE and now - t < CLOCK_SECONDS for cx, cy, t in self.clocks):
                continue
            self.clocks.append((x, y, now))
            if any(abs(p.x - x) < 150 and abs(p.y - y) < 150 and now - p.time < NAME_SECONDS for p in self.pending):
                continue  # the unit itself was seen landing a moment earlier
            if self._spawning(x, y, now):
                continue  # a Skeleton Barrel bursting or a hut spawning drops its units with a timer too
            self.pending.append(Pending(x, y, now, multiplier))
            # react the moment it lands, the name and the track follow a moment later
            self.threat = Threat(UNKNOWN, x, now, y, last_seen=now)
            self.answered = False
        self.clocks = [c for c in self.clocks if now - c[2] < CLOCK_SECONDS]

        for p in [p for p in self.pending if now - p.time >= NAME_SECONDS]:
            self.pending.remove(p)
            self._land(p)
        return self._view(now)

    def _see(self, units: list[kcr.Unit], at: float) -> None:
        """Fold one detector result into the tracks, the board, the open deploys and the threat."""
        current: list[tuple[int, Track]] = []
        for u in units:
            if not u.card:
                continue
            t = self.tracks.setdefault(u.track, Track(first=at, born_enemy=u.enemy))
            t.side += 1 if u.enemy else -1
            t.sightings += 1
            t.cards[u.card] += u.conf
            t.x, t.y, t.last = u.x, u.bottom, at
            current.append((u.track, t))
        self.tracks = {k: t for k, t in self.tracks.items() if at - t.last < 5}

        troops = [(k, t) for k, t in current if not card(t.card).spell]
        self.enemies = [(t.x, t.y) for _, t in troops if t.enemy]
        mine = [(t.card, t.x, t.y) for _, t in troops if not t.enemy]
        board = self.reader.board
        board.my_units = mine
        board.enemy_home = {ln: sum(1 for x, y in self.enemies if lane(x) == ln and y > vision.RIVER_Y) for ln in LANES}
        board.mine_away = {ln: sum(1 for _, x, y in mine if lane(x) == ln and y < vision.RIVER_Y) for ln in LANES}
        board.mine_home = {ln: sum(1 for _, x, y in mine if lane(x) == ln and y >= vision.RIVER_Y) for ln in LANES}
        if self.mine and self.mine.lane is None:
            spotted = [(x, y) for name, x, y in mine if name == self.mine.card]
            if spotted:
                self.mine.lane, self.mine.at = lane(spotted[0][0]), spotted[0]

        # a new enemy unit is a play even when its timer was hidden, behind their king tower for one
        for k, t in troops:
            # decide on its second sighting, a single frame often gets the side wrong
            if t.sightings == 2 and t.side == 2 and t.cards[t.card] >= BIRTH_CONF and not self._known(k, t, t.first):
                self.pending.append(Pending(t.x, t.y, t.first, self.multiplier, source="birth"))
                self.threat = Threat(UNKNOWN, t.x, t.first, t.y, last_seen=at)
                self.answered = False

        # units that appeared on a fresh deploy timer vote on what it was, bystanders do not
        for p in self.pending:
            for k, t in troops:
                if t.first < p.time - VOTE_AGE:
                    continue
                if t.enemy and abs(t.x - p.x) < 70 and -40 < t.y - p.y < 110:
                    p.votes[t.card] += t.cards[t.card] / max(1, sum(t.cards.values()))
                    p.tracks.add(k)
            if self.threat and self.threat.time == p.time and p.votes:
                self.threat.card = p.votes.most_common(1)[0][0]
                self.threat.track = min(p.tracks, key=lambda k: self.tracks[k].first) if p.tracks else None

        # an enemy spell is a play of its own, it never shows a timer, one frame is not enough to trust it
        for k, t in current:
            sure = t.side >= SPELL_SIGHTINGS or (t.first == at and t.born_enemy and t.cards[t.card] >= SURE_SPELL)
            if sure and card(t.card).spell and k not in self.spells_seen and not self._landing_effect(t, at):
                if not any(self.tracks.get(j) and self.tracks[j].card == t.card and at - s < SAME_SPELL_SECONDS
                           for j, s in self.spells_seen.items()):
                    self._land_spell(k, t, at)
                self.spells_seen[k] = at
        # a spell's name settles over its flight, the first frame is often a look-alike
        if any(e.spell and e.track in self.tracks for _, e in self.events):
            for _, e in self.events:
                if e.spell and e.track in self.tracks:
                    e.votes = Counter({c: v for c, v in self.tracks[e.track].cards.items() if card(c).spell})
                    e.named = e.votes.most_common(1)[0][0]
            self._rename()

        # follow the threat by its track, it is gone once the track is
        if self.threat and self.threat.track is not None:
            t = self.tracks.get(self.threat.track)
            if t and at - t.last < THREAT_LOST:
                self.threat.x, self.threat.y, self.threat.last_seen = t.x, t.y, at
            else:
                self.threat = None

    def _landing_effect(self, t: Track, at: float) -> bool:
        """A spell sighting on top of an enemy unit that just landed is that unit's own effect,
        the Ice Wizard's frost burst, the Electro Wizard's zap, the Mega Knight's splash."""
        landings = [(p.x, p.y, p.time) for p in self.pending] + [
            (p.x, p.y, p.time) for p, e in self.events if not e.spell]
        return any(abs(x - t.x) + abs(y - t.y) <= 180 and -0.5 <= at - when <= 1.5 for x, y, when in landings)

    def _spawning(self, x: int, y: int, now: float) -> bool:
        """Whether an enemy card that drops units when it bursts or dies was just here."""
        return any(
            t.enemy and t.card in DEATH_SPAWNERS and now - t.last <= 1.5 and abs(t.x - x) + abs(t.y - y) <= 180
            for t in self.tracks.values()
        )

    def _known(self, k: int, t: Track, at: float) -> bool:
        """Whether a new enemy unit belongs to something already accounted for."""
        for j, other in self.tracks.items():
            if j == k or not other.enemy or other.card is None:
                continue
            gap = at - other.last
            near = abs(other.x - t.x) + abs(other.y - t.y)
            # the tracker lost a unit for a moment and gave it a new id, sometimes under another name
            if other.card == t.card and 0 < gap <= REAPPEAR[0] and near <= REAPPEAR[1]:
                return True
            if 0 < gap <= 1.5 and near <= 120 and not card(other.card).spell:
                return True
            # a spawn from a card already on the arena
            if other.card in SPAWNS.get(t.card, ()) and gap <= 3.0 and near <= SPAWN_RADIUS:
                return True
        # a timer already marked this landing
        if any(abs(p.x - t.x) < 150 and abs(p.y - t.y) < 150 and at - p.time < NAME_SECONDS for p in self.pending):
            return True
        # units from a spell they just cast, a Goblin Barrel's goblins for one
        if any(t.card in SPELL_SPAWNS.get(name, ()) and 0 <= at - when <= SPELL_SPAWN_SECONDS
               for name, when in self.spell_casts):
            return True
        # their units land on their own half, except a few cards and a pocket next to a tower they destroyed
        if t.y > vision.RIVER_Y and t.card not in ANYWHERE:
            pocket = self.reader.board.my_towers.get(lane(t.x)) == 0 and t.y <= POCKET_Y
            if not pocket:
                return True
        return False

    def _track_my_elixir(self, elixir: int, now: float) -> None:
        if elixir < self.my_elixir:
            self.last_drop = now
            if self.threat and now > self.threat.time:
                self.answered = True
        self.my_elixir = elixir

    def _track_my_plays(self, hand: list[str | None], now: float) -> None:
        """A card leaving the hand right after an elixir drop was played."""
        if now - self.last_drop > 1.5:
            return
        for gone in [c for c in self.my_hand if c and c not in hand]:
            self.mine = MyPlay(gone, now)
            self.my_plays.append(self.mine)

    def _land(self, p: Pending) -> None:
        named = p.votes.most_common(1)[0][0] if p.votes else None
        self.events.append((p, Evidence(p.votes, len(p.tracks), None, named=named, source=p.source)))
        self._rename()
        self.played += 1
        self.last_play = self.debug[-1][2]
        if self.threat and self.threat.time == p.time:
            self.threat.card = self.last_play

    def _land_spell(self, track: int, t: Track, when: float) -> None:
        name = t.card
        self.spell_casts.append((name, when))
        p = Pending(t.x, t.y, when, self.multiplier, Counter({name: 2.0}), source="spell")
        self.events.append((p, Evidence(p.votes, 0, None, named=name, spell=True, track=track, source="spell")))
        self.events.sort(key=lambda e: e[0].time)
        self._rename()
        self.played += 1
        self.last_play = name

    def _rename(self) -> None:
        """Best names for every play so far from all evidence."""
        if self.deck_from_api:
            # with their real 8 cards, the cycle lets later plays fix earlier names
            names = decode([e for _, e in self.events], self.opp.deck, self.levels)
        else:
            # each candidate on its own, read through what the detector's classes really mean
            names = [best(e) if e.votes else UNKNOWN for _, e in self.events]
        self._replay(names)

    def _replay(self, names: list[str]) -> None:
        """Rebuild the opponent from the start with the current best names."""
        opp = Opponent(last_time=self.started or 0.0)
        if self.deck_from_api:
            opp.preload(self.opp.deck)
            keep = set(self.opp.deck)
        else:
            # a deck learned from play holds at most 8 cards, the ones the detector backed most
            backing: Counter = Counter()
            for (_, evidence), name in zip(self.events, names):
                backing[name] += sum(evidence.votes.values())
            keep = {n for n, v in backing.most_common(8) if v >= SURE_VOTES and n not in (UNKNOWN, NONE)}
        for (p, _), name in zip(self.events, names):
            if name == NONE:
                continue  # a spawn or an effect, not a play
            name = name if name in keep else UNKNOWN
            played = card(name) if name != UNKNOWN else None
            opp.play(name, played.cost if played else 3, p.time, p.multiplier,
                     champion=bool(played and played.has("champion")))
        self.opp = opp
        self.debug = [(p, dict(p.votes), name) for (p, _), name in zip(self.events, names)]

    def _view(self, now: float) -> View:
        hand = [card(n) if n else None for n in self.my_hand]
        call = advise(Situation(
            now=now, hand=hand, elixir=self.my_elixir, opp=self.opp, threat=self.threat, answered=self.answered,
            board=self.reader.board, mine=self.mine, multiplier=self.multiplier, enemies=self.enemies,
            last_rule=self.last_rule,
        ))
        self.last_rule = call.rule
        return View(
            in_match=True,
            call=call,
            their_hand=self.opp.hand,
            maybe=self.opp.maybe,
            next_card=self.opp.next_card,
            their_elixir=self.opp.elixir,
            my_elixir=self.my_elixir,
            deck_known=self.opp.deck_complete,
            plays=self.opp.plays[-6:],
            queue=self.opp.queue,
            opponent=self.opponent,
            board=self.reader.board,
            my_hand=list(self.my_hand),
        )
