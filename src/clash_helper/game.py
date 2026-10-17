"""One match from frames in to calls out, shared by live play and replays."""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from clash_helper import api, vision
from clash_helper.advisor import Call, Threat, advise
from clash_helper.cards import card
from clash_helper.identify import Detector, name_deploy
from clash_helper.tracker import UNKNOWN, Opponent

VOTE_FRAMES = 8  # frames after a deploy timer that vote on the unit
CLOCK_MERGE = 70  # px, one deploy timer seen over several frames
CLOCK_SECONDS = 1.6
MATCH_GONE_SECONDS = 3.0


@dataclass
class Pending:
    x: int
    y: int
    time: float
    multiplier: float
    votes: Counter = field(default_factory=Counter)
    frames: int = 0
    most_units: int = 0


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


class Match:
    def __init__(self, detector: Detector | None = None, lookup: bool = True) -> None:
        self.detector = detector or Detector()
        self.lookup = lookup
        self._reset()

    def _reset(self) -> None:
        self.opp = Opponent()
        self.started: float | None = None
        self.last_seen = 0.0
        self.pending: list[Pending] = []
        self.clocks: list[tuple[int, int, float]] = []
        self.threat: Threat | None = None
        self.answered = True
        self.my_elixir = 5
        self.looked_up = False
        self.my_hand: list = [None] * 4
        self.played = 0
        self.last_play = ""

    def _lookup_opponent(self, frame: np.ndarray) -> None:
        lines = vision.read_text(frame[95:200, 55:320])
        words = [t for t in lines if not t.replace(" ", "").isdigit()]
        if len(words) < 2:
            return  # no clan on screen, nothing to search by
        name, clan = words[0], words[1]

        def run() -> None:
            try:
                found = api.find_player(name, clan)
            except api.ApiError as error:
                print(f"opponent lookup failed: {error}", flush=True)
                return
            if found and found.get("currentDeck"):
                self.opp.preload([c["name"] for c in found["currentDeck"]])
                print(f"opponent deck: {', '.join(self.opp.deck)}", flush=True)

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
        if self.lookup and not self.looked_up and api.token():
            self.looked_up = True
            self._lookup_opponent(frame)

        multiplier = vision.elixir_multiplier(frame)
        self.opp.tick(now, multiplier)
        self._track_my_elixir(vision.own_elixir(frame), now)

        for x, y in vision.deploy_clocks(frame):
            if y > vision.ARENA_BOTTOM - 120:
                continue  # emote bubbles over the king tower look like timers
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
            near_badges = [b for b in badges if abs(b[0] - p.x) < 110 and -80 < b[1] - p.y < 140]
            p.most_units = max(p.most_units, len(near_badges))
            for name, conf, x, top, bottom in units:
                if not (abs(x - p.x) < 90 and -60 < bottom - p.y < 110):
                    continue
                # enemy units wear their crimson level badge just above the head
                if any(abs(bx - x) < 40 and top - 45 < by < top + 35 for bx, by in badges):
                    p.votes[name] += conf
        for p in [p for p in self.pending if p.frames >= VOTE_FRAMES]:
            self.pending.remove(p)
            self._land(p)

    def _land(self, p: Pending) -> None:
        candidates = self.opp.deck if self.opp.deck_complete else None
        if candidates and self.opp.hand:
            # a complete cycle says which 4 cards they could have played
            candidates = self.opp.hand + self.opp.maybe
        name = name_deploy(p.votes, candidates, p.most_units) or UNKNOWN
        played = card(name) if name != UNKNOWN else None
        cost = played.cost if played else 3
        self.opp.play(name, cost, p.time, p.multiplier, champion=bool(played and played.has("champion")))
        self.played += 1
        self.last_play = name
        if played and not played.spell and not played.building:
            self.threat = Threat(name, p.x, p.time)
            self.answered = False

    def _view(self, now: float) -> View:
        call = advise(now, self.my_hand, self.my_elixir, self.opp, self.threat, self.answered)
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
        )
