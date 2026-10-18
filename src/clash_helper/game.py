"""One match from frames in to calls out, shared by live play and replays.

The KataCR detector sees every unit and spell with its side and a track id.
The red deploy timer says exactly when and where an enemy card lands, the
tracked units around it say what it was and follow it down the lane.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from clash_helper import api, bots, confusion, kcr, vision
from clash_helper.advisor import Call, MyPlay, Situation, Threat, advise
from clash_helper.board import LANES, Board, BoardReader, lane
from clash_helper.cards import card
from clash_helper.decode import NONE, Evidence, best, decode, expected_cost
from clash_helper.hand import HandReader
from clash_helper.tracker import UNKNOWN, Opponent

NAME_SECONDS = 1.4  # how long after a deploy timer the units around it vote on the card
CLOCK_MERGE = 70  # px, one deploy timer seen over several frames
CLOCK_SECONDS = 2.2  # a timer shows about this long counted from the detector's first sight of it
MATCH_GONE_SECONDS = 3.0
REGULATION = 180  # seconds on the clock when a match starts
THREAT_LOST = 1.2  # seconds a tracked unit can go unseen before it counts as dead
SAME_SPELL_SECONDS = 2.0
SURE_SPELL = 0.9  # a spell seen only where it lands counts once one detection is this confident
SPELL_TRAVEL = 80  # px a spell must roll or fly before its direction tells whose it is, explosions jitter less
# spells the detector follows across the arena, the detector's side guess for them is no help but their way is
ROLLING = {"The Log", "Barbarian Barrel", "Goblin Barrel"}
CHAIN = (1.0, 250)  # seconds and px within which a spell the tracker lost and found again is the same cast
BIRTH_CONF = 0.6  # a new enemy unit this confident is a play even without a visible timer
BIRTH_SIGHTINGS = 4  # its first sightings that may settle its side, single frames get it wrong
LATE_UNIT = 3.5  # seconds after its timer a slow unit may still show up, behind their king tower for one
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
# signal thresholds from research/spell_signals.py, with the card in their deck and without
BOLT = (0.4, 0.9)  # share of the top strip a bolt from the sky lights up
BOLT_CARDS = {"Zap", "Electro Wizard"}  # both drop in as a bolt, a unit left under it tells which
BOLT_Y = 500  # where a bolt's landing is assumed until a unit shows the spot
ARROWS_RING = (0.25, 0.45)  # strength of the salmon impact ring
ARROWS_FLIGHT = 0.6  # seconds from launch to the ring
CUE_SECONDS = 2.5  # one cast, seen over several frames or also by the detector
VOTE_AGE = 0.5  # only units that appeared around the landing vote, not bystanders already there
ANYWHERE = {"Miner", "Goblin Drill", "Royal Delivery"}  # land on your side without a destroyed tower
SURE_VOTES = 1.2  # summed confidence before a name may join a deck learned from play
UNNAMED_COST = 3  # elixir assumed for a play that has no name yet
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
    crowd: int = 0  # most units seen landing here at once, the tracker's new ids for one unit add up otherwise
    source: str = "timer"  # timer, birth, bolt or spell

    def covers(self, x: int, y: int) -> bool:
        """Whether a spot belongs to this landing, a bolt from the sky only gives its column."""
        return abs(self.x - x) < 150 and (self.source == "bolt" or abs(self.y - y) < 150)


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
    x0: int = 0  # where it was first seen, a spell's direction of travel gives away its side
    y0: int = 0
    peak: float = 0.0  # most confident single detection
    judged: bool = False  # whether it was already weighed as a new enemy play
    since: dict[str, float] = field(default_factory=dict)  # when each class first showed, a new unit can take an old id

    @property
    def flying_down(self) -> bool | None:
        """For a moving spell, whether it travels down the screen, from their side toward yours."""
        if abs(self.y - self.y0) < SPELL_TRAVEL:
            return None
        return self.y > self.y0

    @property
    def enemy(self) -> bool:
        return self.side > 0

    @property
    def card(self) -> str | None:
        return self.cards.most_common(1)[0][0] if self.cards else None


@dataclass
class Observation:
    """What one frame shows, cached by the research tools so experiments skip the video."""

    now: float
    in_match: bool
    time_left: int | None = None  # read only while the match start is still unknown
    multiplier: float = 1.0
    elixir: int = 0
    hand: list[str | None] | None = None
    clocks: list[tuple[int, int]] = field(default_factory=list)
    units_at: float = 0.0
    units: list = field(default_factory=list)
    towers: dict | None = None  # raw health reads, when this frame was read
    bolt: tuple[float, int] = (0.0, 0)  # a Zap falling through the top strip, share and x
    ring: tuple[float, int, int] = (0.0, 0, 0)  # an enemy Arrows impact ring, score and center


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
        self.guess: list[str] | None = None  # their deck worked out from play, all 8 seen or a bot's preset
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
        self.last_sky = None
        self.first_frame = 0.0
        self.last_rule = ""
        self.enemies: list[tuple[int, int]] = []
        self.tracks: dict[int, Track] = {}
        self.alias: dict[int, int] = {}  # track ids the tracker reused or changed -> the track they belong to
        self.next_key = 0  # keys for tracks split off a reused id count down from here
        self.last_bolt = -10.0
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
        return self.update(self.observe(frame, now))

    def observe(self, frame: np.ndarray, now: float) -> Observation:
        """Everything this frame shows, the only part that needs the image."""
        obs = Observation(now, vision.in_match(frame))
        if not obs.in_match:
            return obs
        if self.started is None:
            obs.time_left = vision.time_left(frame)
        if self.lookup and self.lookup_state == "todo" and api.token():
            self._lookup_opponent(frame, now)
        obs.multiplier = vision.elixir_multiplier(frame)
        obs.elixir = vision.own_elixir(frame)
        if self.hand_reader:
            obs.hand = self.hand_reader.read(frame)[0]
        # the detector works in the background, the newest result rides along
        self.detector.submit(frame, now)
        while self.sync and self.detector.busy:
            time.sleep(0.002)
        obs.units_at, obs.units = self.detector.result()
        obs.clocks = vision.deploy_clocks(frame)
        sky = vision.sky(frame)
        if self.last_sky is not None:
            obs.bolt = vision.sky_bolt(sky, self.last_sky)
        self.last_sky = sky
        obs.ring = vision.arrows_ring(frame)
        self.reader.towers(frame, now)
        return obs

    def update(self, obs: Observation) -> View:
        """Advance the match by one observation, live frames and cached replays alike."""
        now = obs.now
        if not obs.in_match:
            if self.started is not None and now - self.last_seen > MATCH_GONE_SECONDS:
                # keep what happened for the journal, the next match starts clean
                self.ended = {"plays": list(self.debug), "opponent": self.opponent, "deck": list(self.opp.deck),
                              "lookup": self.lookup_note, "deck_from_api": self.deck_from_api,
                              "deck_guessed": bool(self.guess)}
                self._reset()
            return View(in_match=False) if self.started is None else self._view(now)
        if self.started is None:
            self.first_frame = now
            # the helper often notices a match a few seconds in, the clock says how many
            left = obs.time_left
            elapsed = REGULATION - left if left is not None and 0 < left <= REGULATION else 0
            self.started = now - elapsed
            self.opp.last_time = self.started
            threading.Thread(target=self._load_my_deck, daemon=True).start()
        self.last_seen = now

        multiplier = self.multiplier = obs.multiplier
        self.opp.tick(now, multiplier)
        self._track_my_elixir(obs.elixir, now)
        if obs.hand is not None:
            # a card mid-slide or being dragged reads as None, keep the last good name
            hand = [new or old for new, old in zip(obs.hand, self.my_hand)]
            self._track_my_plays(hand, now)
            self.my_hand = hand
        if obs.towers is not None:
            self.reader.apply(obs.towers)
        if obs.units_at > self.last_seen_units:
            self.last_seen_units = obs.units_at
            self._see(obs.units, obs.units_at)

        for x, y in obs.clocks:
            self._timer(x, y, now)
        self.clocks = [c for c in self.clocks if now - c[2] < CLOCK_SECONDS]

        # cards the detector misses, by their own marks on the screen
        deck = set(self.opp.deck) if self.deck_from_api else set(self.guess or self._learned_deck(self._names()))
        settled = now - self.first_frame > 1.0  # the first comparison of a match flashes on its own
        share, x = obs.bolt if settled else (0.0, 0)
        if share >= self._bar(BOLT_CARDS, BOLT, deck) and now - self.last_bolt > CUE_SECONDS:
            self.last_bolt = now
            # a unit landing high up pokes into the strip too, its own timer already counts it
            if not any(abs(p.x - x) < 100 and now - p.time < NAME_SECONDS for p in self.pending):
                self.pending.append(Pending(x, BOLT_Y, now, multiplier, source="bolt"))
        strength, x, y = obs.ring if settled else (0.0, 0, 0)
        if strength >= self._bar({"Arrows"}, ARROWS_RING, deck):
            # the ring shows at impact, the volley left their tower a little earlier
            self._land_cue("Arrows", x, y, now - ARROWS_FLIGHT)

        for p in [p for p in self.pending if now - p.time >= NAME_SECONDS]:
            self.pending.remove(p)
            self._land(p)
        return self._view(now)

    def _timer(self, x: int, y: int, now: float) -> None:
        """An enemy deploy timer, the first sign of a card landing there."""
        if y > vision.ARENA_BOTTOM - 120 or x > 615:
            return  # emote bubbles and the crown counter look like timers
        if any(abs(x - cx) + abs(y - cy) < CLOCK_MERGE and now - t < CLOCK_SECONDS for cx, cy, t in self.clocks):
            return  # the same timer seen again
        self.clocks.append((x, y, now))
        if any(p.covers(x, y) and now - p.time < NAME_SECONDS for p in self.pending):
            return  # the unit itself was seen landing a moment earlier
        if self._spawning(x, y, now):
            return  # a Skeleton Barrel bursting or a hut spawning drops its units with a timer too
        self.pending.append(Pending(x, y, now, self.multiplier))
        # react the moment it lands, the name and the track follow a moment later
        self.threat = Threat(UNKNOWN, x, now, y, last_seen=now)
        self.answered = False

    def _bar(self, cards: set[str], bars: tuple[float, float], deck: set[str]) -> float:
        """How strong a mark must be to count. The marks also flash for other reasons, so a weak one
        counts when their deck holds one of the cards and none does when their known or guessed deck holds none."""
        if cards & deck:
            return bars[0]
        return math.inf if self.deck_from_api or self.guess else bars[1]

    def _land_cue(self, name: str, x: int, y: int, when: float) -> None:
        """A spell found by its mark rather than the detector, once per cast and never your own."""
        if any(n == name and abs(when - t) < CUE_SECONDS for n, t in self.spell_casts):
            return
        if any(p.card == name and abs(when - p.time) < CUE_SECONDS for p in self.my_plays):
            return
        self.spell_casts.append((name, when))
        p = Pending(x, y, when, self.multiplier, Counter({confusion.CUE + name: 2.0}), source="spell")
        self.events.append((p, Evidence(p.votes, 0, None, named=name, spell=True, source="spell")))
        self.events.sort(key=lambda e: e[0].time)
        self._rename()
        self.played += 1
        self.last_play = name

    def _see(self, units: list[kcr.Unit], at: float) -> None:
        """Fold one detector result into the tracks, the board, the open deploys and the threat."""
        self.reader.see(units, at)
        # the detector spots a deploy timer while its red wedge is still too thin for the screen check,
        # it now and then takes one of yours for theirs, but those land on your half or in your pocket
        for u in units:
            y = (u.top + u.bottom) // 2
            if u.name == "clock" and u.enemy and y <= vision.RIVER_Y + 20 and not self._my_pocket(u.x, at):
                self._timer(u.x, y, at)

        current: list[tuple[int, Track]] = []
        present = {self.alias.get(u.track, u.track) for u in units}
        for u in units:
            if not u.card:
                continue
            k = self._key(u, at, present)
            t = self.tracks.setdefault(k, Track(first=at, born_enemy=u.enemy, x0=u.x, y0=u.bottom))
            t.side += 1 if u.enemy else -1
            t.sightings += 1
            t.cards[u.card] += u.conf
            t.peak = max(t.peak, u.conf)
            t.since.setdefault(u.card, at)
            t.x, t.y, t.last = u.x, u.bottom, at
            current.append((k, t))
        self.tracks = {k: t for k, t in self.tracks.items() if at - t.last < 5}
        self.alias = {a: k for a, k in self.alias.items() if k in self.tracks}

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
            # decide once its first sightings lean their way, a single frame often gets the side wrong
            if t.judged or t.sightings > BIRTH_SIGHTINGS or t.side < 2 or t.cards[t.card] < BIRTH_CONF:
                continue
            t.judged = True
            named = self._named_landing(t, at)
            if named is not None:
                # a slow unit showing up on a timer that was already named, it says what that play was
                named.votes[t.card] += t.cards[t.card] / max(1, sum(t.cards.values()))
                self._rename()
            elif not self._known(k, t, t.first):
                self.pending.append(Pending(t.x, t.y, t.first, self.multiplier, source="birth"))
                self.threat = Threat(UNKNOWN, t.x, t.first, t.y, last_seen=at)
                self.answered = False

        # units that appeared on a fresh deploy timer vote on what it was, bystanders do not. A new
        # unit sometimes takes over the track of one that just left, so it is the class that must be new
        for p in self.pending:
            here = 0
            for k, t in troops:
                if not t.enemy or abs(t.x - p.x) >= 70 or not (p.source == "bolt" or -40 < t.y - p.y < 110):
                    continue
                # an old unit's new class must hold for a couple of frames, a dying unit flickers
                fresh = t.first >= p.time - VOTE_AGE
                new = {c: v for c, v in t.cards.items() if t.since[c] >= p.time - VOTE_AGE and (fresh or v >= 1.0)}
                if new:
                    here += 1
                    if p.source == "bolt" and not p.votes:
                        p.x, p.y = t.x, t.y  # the first unit under a bolt shows where it landed
                    top = max(new, key=new.get)
                    p.votes[top] += new[top] / max(1, sum(new.values()))
                    p.tracks.add(k)
            p.crowd = max(p.crowd, here)
            if p.source == "bolt" and p.votes and (self.threat is None or self.threat.time < p.time):
                self.threat = Threat(UNKNOWN, p.x, p.time, p.y, last_seen=at)
                self.answered = False
            if self.threat and self.threat.time == p.time and p.votes:
                self.threat.card = p.votes.most_common(1)[0][0]
                self.threat.track = min(p.tracks, key=lambda k: self.tracks[k].first) if p.tracks else None

        # an enemy spell is a play of its own, it never shows a timer
        for k, t in current:
            if not card(t.card).spell:
                continue
            # their spells roll and fly down toward you and yours up, the detector's side guess for spells
            # is no help. One seen only where it lands is theirs unless you just cast it
            direction = t.flying_down
            if direction is not None:
                sure = direction
            else:
                sure = t.card not in ROLLING and t.peak >= SURE_SPELL and not self._cast_by_me(t.card, t.first)
            if sure and k not in self.spells_seen and not self._landing_effect(t.x, t.y, at):
                if not any(self.tracks.get(j) and self.tracks[j].card == t.card and at - s < SAME_SPELL_SECONDS
                           for j, s in self.spells_seen.items()):
                    self._land_spell(k, t, t.first)  # they paid when it left their hand
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

    def _key(self, u: kcr.Unit, at: float, present: set[int]) -> int:
        """The track a detection belongs to. The tracker loses a spell for a moment and finds it again
        under a new id, and hands a lost unit's id to a spell or the other way round."""
        k = self.alias.get(u.track, u.track)
        t = self.tracks.get(k)
        spell = card(u.card).spell
        if t is not None and card(t.card).spell == spell:
            return k
        if spell:
            for j, other in self.tracks.items():
                if j in present or other.card != u.card or not 0 < at - other.last <= CHAIN[0]:
                    continue
                if abs(other.x - u.x) + abs(other.y - u.bottom) <= CHAIN[1]:
                    self.alias[u.track] = j
                    return j
        if t is None:
            return k
        self.next_key -= 1
        self.alias[u.track] = self.next_key
        return self.next_key

    def _named_landing(self, t: Track, at: float) -> Pending | None:
        """A timer landing right where a new unit stands that was named a moment ago."""
        return next((p for p, e in reversed(self.events) if not e.spell and p.source in ("timer", "bolt")
                     and 0 <= at - p.time <= LATE_UNIT and p.covers(t.x, t.y)), None)

    def _my_pocket(self, x: int, at: float) -> bool:
        """Whether you may have just deployed there, past the river where you took their tower."""
        return self.reader.board.their_towers.get(lane(x)) == 0 and at - self.last_drop < 1.5

    def _cast_by_me(self, name: str, when: float) -> bool:
        return any(p.card == name and -1.0 <= when - p.time <= 4.0 for p in self.my_plays)

    def _landing_effect(self, x: int, y: int, at: float) -> bool:
        """A spell seen on top of an enemy unit that just landed is that unit's own effect,
        the Ice Wizard's frost burst, the Electro Wizard's zap, the Mega Knight's splash."""
        landings = [*self.pending, *(p for p, e in self.events if not e.spell)]
        return any(-0.5 <= at - p.time <= 1.5 and abs(p.x - x) + (0 if p.source == "bolt" else abs(p.y - y)) <= 180
                   for p in landings)

    def _spawning(self, x: int, y: int, now: float) -> bool:
        """Whether an enemy card that drops units when it bursts or dies was just here."""
        return any(
            t.enemy and t.card in DEATH_SPAWNERS and self._theirs(t) and now - t.last <= 1.5
            and abs(t.x - x) + abs(t.y - y) <= 180
            for t in self.tracks.values()
        )

    def _theirs(self, t: Track) -> bool:
        """Whether a track is surely one of their cards, seen a few times and in their deck when it is known."""
        return t.sightings >= 3 and (not self.deck_from_api or t.card in self.opp.deck)

    def _known(self, k: int, t: Track, at: float) -> bool:
        """Whether a new enemy unit belongs to something already accounted for."""
        for j, other in self.tracks.items():
            if j == k or other.card is None:
                continue
            gap = at - other.last
            near = abs(other.x - t.x) + abs(other.y - t.y)
            # the tracker lost a unit for a moment or doubled it under a new id, its side guess can flip too
            if other.card == t.card and 0 <= gap <= REAPPEAR[0] and near <= REAPPEAR[1]:
                return True
            if not other.enemy:
                continue
            # or gave it a new id under another name
            if 0 < gap <= 1.5 and near <= 120 and not card(other.card).spell:
                return True
            # a spawn from a card already on the arena
            if other.card in SPAWNS.get(t.card, ()) and self._theirs(other) and gap <= 3.0 and near <= SPAWN_RADIUS:
                return True
        # a timer already marked this landing
        if any(p.covers(t.x, t.y) and at - p.time < NAME_SECONDS for p in self.pending):
            return True
        # a card you just played, its side misread
        if self._cast_by_me(t.card, at):
            return True
        # units from a spell they just cast, a Goblin Barrel's goblins for one
        if any(t.card in SPELL_SPAWNS.get(name, ()) and 0 <= at - when <= SPELL_SPAWN_SECONDS
               for name, when in self.spell_casts):
            return True
        # their units land on their own half except a few cards. Next to a tower they destroyed a deploy
        # shows its timer in plain view, a new unit there without one is yours with its side misread
        return t.y > vision.RIVER_Y and t.card not in ANYWHERE

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
            # the card leaves the hand once the next one slides in, the elixir dropped when it was played
            self.mine = MyPlay(gone, self.last_drop)
            self.my_plays.append(self.mine)

    def _land(self, p: Pending) -> None:
        if p.source == "bolt" and not p.votes:
            self._land_cue("Zap", p.x, p.y, p.time)  # nothing landed under the bolt
            return
        named = p.votes.most_common(1)[0][0] if p.votes else None
        self.events.append((p, Evidence(p.votes, p.crowd, None, named=named, source=p.source)))
        self._rename()
        self.played += 1
        self.last_play = self.debug[-1][2]
        if self.threat and self.threat.time == p.time:
            self.threat.card = self.last_play

    def _land_spell(self, track: int, t: Track, when: float) -> None:
        name = t.card
        if any(n == name and abs(when - w) < CUE_SECONDS for n, w in self.spell_casts):
            return  # its mark on the screen already counted it
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
            learned = self._learned_deck(names)
            # a bot's preset deck that holds the cards they have shown, or all 8 of their cards seen.
            # Worked out again on every new play, cards outside the guess drop it
            self.guess = bots.deck(frozenset(learned)) or (sorted(learned) if len(learned) == 8 else None)
            if self.guess:
                # the cycle helps exactly as with a looked up deck
                names = decode([e for _, e in self.events], self.guess, {})
        self._replay(names)

    def _names(self) -> list[str]:
        return [name for _, _, name in self.debug]

    def _learned_deck(self, names: list[str]) -> set[str]:
        """Up to 8 cards their plays most strongly point to."""
        backing: Counter = Counter()
        for (_, evidence), name in zip(self.events, names):
            backing[name] += sum(evidence.votes.values())
        return {n for n, v in backing.most_common(8) if v >= SURE_VOTES and n not in (UNKNOWN, NONE)}

    def _replay(self, names: list[str]) -> None:
        """Rebuild the opponent from the start with the current best names."""
        opp = Opponent(last_time=self.started or 0.0)
        deck = self.opp.deck if self.deck_from_api else self.guess
        if deck:
            opp.preload(deck)
            keep = set(deck)
        else:
            # a deck learned from play holds at most 8 cards, the ones the detector backed most
            keep = self._learned_deck(names)
        for (p, evidence), name in zip(self.events, names):
            if name == NONE:
                continue  # a spawn or an effect, not a play
            name = name if name in keep else UNKNOWN
            played = card(name) if name != UNKNOWN else None
            # their looked up deck and the cycle pin a name down, otherwise a costly card read with
            # little confidence would throw their elixir far off
            if played and self.deck_from_api:
                cost = played.cost
            else:
                cost = expected_cost(evidence, deck, self.levels) if evidence.votes else UNNAMED_COST
            opp.play(name, cost, p.time, p.multiplier, champion=bool(played and played.has("champion")))
        self.opp = opp
        self.debug = [(p, dict(p.votes), name) for (p, _), name in zip(self.events, names)]

    @property
    def their_elixir(self) -> float:
        """Their elixir with the cards still landing already paid for, at what their hand costs on average."""
        if not self.pending:
            return self.opp.elixir
        hand = [card(n).cost for n in self.opp.hand]
        each = sum(hand) / len(hand) if hand else UNNAMED_COST
        return max(0.0, self.opp.elixir - each * len(self.pending))

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
            their_elixir=self.their_elixir,
            my_elixir=self.my_elixir,
            deck_known=self.opp.deck_complete,
            plays=self.opp.plays[-6:],
            queue=self.opp.queue,
            opponent=self.opponent,
            board=self.reader.board,
            my_hand=list(self.my_hand),
        )
