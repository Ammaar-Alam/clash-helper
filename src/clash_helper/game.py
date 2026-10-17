"""One match from frames in to calls out, shared by live play and replays."""

from __future__ import annotations

import queue
import threading
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from clash_helper import api, namer, vision
from clash_helper.advisor import Call, Threat, advise
from clash_helper.cards import all_cards, card
from clash_helper.decode import Evidence, decode
from clash_helper.hand import HandReader
from clash_helper.identify import Detector, name_deploy
from clash_helper.tracker import UNKNOWN, Opponent

VOTE_FRAMES = 8  # frames after a deploy timer that count badges
SPAWN_FRAMES = 5  # of those, frames where the new unit is still on its timer
CLOCK_MERGE = 70  # px, one deploy timer seen over several frames
CLOCK_SECONDS = 1.6
MATCH_GONE_SECONDS = 3.0
WIDE_BADGE = 18  # px, a level of 10 or more
LOOKUP_EVERY = 1.0  # seconds between name reads while the intro fades
LOOKUP_WINDOW = 25.0


@dataclass
class Pending:
    x: int
    y: int
    time: float
    multiplier: float
    votes: Counter = field(default_factory=Counter)
    frames: int = 0
    counts: list[int] = field(default_factory=list)
    widths: list[int] = field(default_factory=list)
    shots: list[np.ndarray] = field(default_factory=list)  # frames 0.4 s apart for the vision model

    @property
    def units(self) -> int:
        """Badge count around the landing spot once everything dropped in, robust to passers-by."""
        if not self.counts:
            return 0
        return sorted(self.counts)[int(0.75 * (len(self.counts) - 1))]

    @property
    def digits(self) -> int | None:
        """How many digits the units' level has, a two digit badge is clearly wider."""
        if not self.widths:
            return None
        return 2 if max(self.widths) >= WIDE_BADGE else 1


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


class Match:
    def __init__(self, detector: Detector | None = None, lookup: bool = True) -> None:
        self.detector = detector or Detector()
        self.lookup = lookup
        self.hand_reader: HandReader | None = None
        self._reset()
        tag = api.setting("CR_PLAYER_TAG")
        if lookup and tag and api.token():
            threading.Thread(target=self._load_my_deck, args=(tag,), daemon=True).start()

    def _load_my_deck(self, tag: str) -> None:
        try:
            self.hand_reader = HandReader(api.player(tag)["currentDeck"])
        except (api.ApiError, OSError) as error:
            print(f"could not load your deck: {error}", flush=True)

    def _reset(self) -> None:
        self.opp = Opponent()
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
        self.last_ocr = -10.0
        self.my_hand: list = [None] * 4
        self.next_mine: str | None = None
        self.played = 0
        self.events: list[tuple[Pending, Evidence]] = []
        self.answers: queue.Queue[tuple[Pending, str | None]] = queue.Queue()
        self.debug: list = []
        self.last_play = ""

    def _lookup_opponent(self, frame: np.ndarray, now: float) -> None:
        """Read their name and clan, then fetch the deck they are using right now."""
        if now - self.last_ocr < LOOKUP_EVERY or now - (self.started or now) > LOOKUP_WINDOW:
            return
        self.last_ocr = now
        # name, clan, trophies, the clan line is missing for clanless players
        lines = [t for t in vision.read_text(frame[95:200, 55:320]) if not t.replace(" ", "").isdigit()]
        if len(lines) < 2:
            return
        name, clan = lines[0], lines[1]
        self.opponent = name
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
            if found and found.get("currentDeck"):
                self.opp.preload([c["name"] for c in found["currentDeck"]])
                # the badge shows the level counted from 16 down for every rarity
                self.levels = {c["name"]: c["level"] + 16 - c["maxLevel"] for c in found["currentDeck"]}
                print(f"opponent deck: {', '.join(self.opp.deck)}", flush=True)
            self.lookup_state = "done"

        threading.Thread(target=run, daemon=True).start()

    def step(self, frame: np.ndarray, now: float) -> View:
        if not vision.in_match(frame):
            if self.started is not None and now - self.last_seen > MATCH_GONE_SECONDS:
                self._reset()
            return View(in_match=False) if self.started is None else self._view(now)
        if self.started is None:
            self.started = now
            self.opp.last_time = now
        self.last_seen = now
        if self.lookup and self.lookup_state == "todo" and api.token():
            self._lookup_opponent(frame, now)

        self._take_answers()
        multiplier = vision.elixir_multiplier(frame)
        self.opp.tick(now, multiplier)
        self._track_my_elixir(vision.own_elixir(frame), now)
        if self.hand_reader:
            hand, upcoming = self.hand_reader.read(frame)
            # a card mid-slide or being dragged reads as None, keep the last good name
            self.my_hand = [new or old for new, old in zip(hand, self.my_hand)]
            self.next_mine = upcoming or self.next_mine

        for x, y in vision.deploy_clocks(frame):
            if y > vision.ARENA_BOTTOM - 120 or x > 615:
                continue  # emote bubbles and the crown counter look like timers
            if any(abs(x - cx) + abs(y - cy) < CLOCK_MERGE and now - t < CLOCK_SECONDS for cx, cy, t in self.clocks):
                continue
            self.clocks.append((x, y, now))
            self.pending.append(Pending(x, y, now, multiplier))
        self.clocks = [c for c in self.clocks if now - c[2] < CLOCK_SECONDS]

        if self.pending:
            self._vote(frame)
        return self._view(now)

    def _track_my_elixir(self, elixir: int, now: float) -> None:
        if elixir < self.my_elixir and self.threat and now > self.threat.time:
            self.answered = True
        self.my_elixir = elixir

    def _vote(self, frame: np.ndarray) -> None:
        units = self.detector.detect(frame)
        badges = vision.level_badges(frame, enemy=True)
        for p in self.pending:
            p.frames += 1
            if p.frames % 2 == 1:
                p.shots.append(frame)
            if p.frames >= 2:
                # the units need a moment to drop in and show their badges
                near = [w for bx, by, w in badges if abs(bx - p.x) < 75 and -90 < by - p.y < 60]
                p.counts.append(len(near))
                p.widths += near
            for name, conf, x, top, bottom in units:
                # a fresh unit stands on its deploy timer, older units nearby have walked off it
                if p.frames > SPAWN_FRAMES or not (abs(x - p.x) < 50 and -30 < bottom - p.y < 80):
                    continue
                # enemy units wear their crimson level badge just above the head
                if any(abs(bx - x) < 40 and top - 45 < by < top + 35 for bx, by, _ in badges):
                    p.votes[name] += conf
        for p in [p for p in self.pending if p.frames >= VOTE_FRAMES]:
            self.pending.remove(p)
            self._land(p)

    def _land(self, p: Pending) -> None:
        evidence = Evidence(p.votes, p.units, p.digits)
        self.events.append((p, evidence))
        self._rename()
        self.played += 1
        self.last_play = self.debug[-1][2]
        played = card(self.last_play) if self.last_play != UNKNOWN else None
        if played and not played.spell and not played.building:
            self.threat = Threat(self.last_play, p.x, p.time)
            self.answered = False
        if api.setting("GEMINI_API_KEY") and p.shots:
            if self.opp.deck_complete:
                options = {n: self.levels.get(n) for n in self.opp.deck}
            else:
                options = {n: None for n, c in all_cards().items() if not c.spell}
            image = namer.strip(p.shots, p.x, p.y)
            threading.Thread(target=lambda: self.answers.put((p, namer.name_card(image, options))), daemon=True).start()

    def _take_answers(self) -> None:
        changed = False
        while not self.answers.empty():
            p, answer = self.answers.get_nowait()
            for q, evidence in self.events:
                if q is p and answer:
                    evidence.named = answer
                    changed = True
        if changed:
            self._rename()
            if self.threat and self.events and self.threat.time == self.events[-1][0].time:
                self.threat = Threat(self.debug[-1][2], self.threat.x, self.threat.time)

    def _rename(self) -> None:
        """Best names for every play so far from all evidence."""
        if self.opp.deck_complete:
            # with their 8 cards known, the cycle lets later plays fix earlier names
            names = decode([e for _, e in self.events], self.opp.deck, self.levels)
        else:
            names = [
                e.named or name_deploy(e.votes, None, e.units) or UNKNOWN
                for _, e in self.events
            ]
        self._replay(names)

    def _replay(self, names: list[str]) -> None:
        """Rebuild the opponent from the start with the current best names."""
        opp = Opponent(last_time=self.started or 0.0)
        if self.opp.deck_complete:
            opp.preload(self.opp.deck)
        for (p, _), name in zip(self.events, names):
            played = card(name) if name != UNKNOWN else None
            opp.play(name, played.cost if played else 3, p.time, p.multiplier, champion=bool(played and played.has("champion")))
        self.opp = opp
        self.debug = [(p, dict(p.votes), name) for (p, _), name in zip(self.events, names)]

    def _view(self, now: float) -> View:
        hand = [card(n) if n else None for n in self.my_hand]
        call = advise(now, hand, self.my_elixir, self.opp, self.threat, self.answered)
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
        )
