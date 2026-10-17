"""Record every match with every call so the rules can be measured over many games.

Each match gets a folder with a small video, a line per event and a summary
that picks up the result from the battle log once the game is over.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from clash_helper import api

HOME = Path.home() / "Library/Application Support/clash-helper"
GAMES = HOME / "games"
SETTINGS = HOME / "settings.json"
VIDEO_SIZE = (664, 1440)  # the full canonical screen, sharp enough to train on
STATE_EVERY = 1.0


def recording() -> bool:
    try:
        return bool(json.loads(SETTINGS.read_text()).get("record", True))
    except (OSError, json.JSONDecodeError):
        return True


def set_recording(on: bool) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps({"record": on}))


class Journal:
    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.folder: Path | None = None

    def _start(self, now: float) -> None:
        self.folder = GAMES / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.video = cv2.VideoWriter(str(self.folder / "match.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, VIDEO_SIZE)
        self.events = (self.folder / "events.jsonl").open("w")
        self.started_at = datetime.now(timezone.utc)
        self.start = now
        self.last_call = ""
        self.last_state = -STATE_EVERY
        self.mine_logged = 0

    def _log(self, now: float, kind: str, **data) -> None:
        self.events.write(json.dumps({"t": round(now - self.start, 2), "type": kind, **data}) + "\n")

    def update(self, match, frame: np.ndarray, view, now: float) -> None:
        """Feed every frame, `match` is the game.Match producing `view`."""
        if view.in_match and self.folder is None:
            if not recording():
                return
            self._start(now)
        if self.folder is None:
            return
        if not view.in_match:
            self._finish(match)
            return
        self.video.write(cv2.resize(frame, VIDEO_SIZE, interpolation=cv2.INTER_AREA))
        call = asdict(view.call)
        if str(view.call) != self.last_call:
            self.last_call = str(view.call)
            self._log(now, "call", **call)
        for play in match.my_plays[self.mine_logged:]:
            self._log(play.time, "my_play", card=play.card, lane=play.lane)
        self.mine_logged = len(match.my_plays)
        if now - self.last_state >= STATE_EVERY:
            self.last_state = now
            board = view.board.snapshot() if view.board else {}
            self._log(now, "state", my_elixir=view.my_elixir, their_elixir=round(view.their_elixir, 1),
                      my_hand=view.my_hand, their_hand=view.their_hand, **board)

    def _finish(self, match) -> None:
        self.video.release()
        ended = match.ended
        theirs = [{"t": round(p.time - self.start, 2), "card": name, "x": p.x, "y": p.y} for p, _, name in ended.get("plays", [])]
        for play in theirs:
            self._log(self.start + play["t"], "their_play", **play)
        self.events.close()
        summary = {
            "started": self.started_at.isoformat(),
            "opponent": ended.get("opponent", ""),
            "their_deck": ended.get("deck", []),
            "deck_from_api": ended.get("deck_from_api", False),
            "lookup": ended.get("lookup", ""),
            "my_deck": [c["name"] for c in match.my_deck] if match.my_deck else [],
            "result": None,
        }
        folder = self.folder
        (folder / "summary.json").write_text(json.dumps(summary, indent=1))
        self.folder = None
        threading.Thread(target=self._result, args=(folder, summary), daemon=True).start()

    def _result(self, folder: Path, summary: dict) -> None:
        """The battle log shows a finished game within a minute or so."""
        tag = api.setting("CR_PLAYER_TAG")
        if not tag or not api.token():
            return
        started = datetime.fromisoformat(summary["started"])
        for _ in range(8):
            time.sleep(15)
            try:
                battles = api.battlelog(tag)
            except api.ApiError:
                continue
            for b in battles:
                when = datetime.strptime(b["battleTime"], "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)
                if when >= started:
                    me, them = b["team"][0], b["opponent"][0]
                    summary["result"] = {
                        "won": me["crowns"] > them["crowns"],
                        "crowns": [me["crowns"], them["crowns"]],
                        "opponent": them["name"],
                        "their_deck": [c["name"] for c in them["cards"]],
                        "my_deck": [c["name"] for c in me["cards"]],
                        "trophy_change": me.get("trophyChange"),
                    }
                    (folder / "summary.json").write_text(json.dumps(summary, indent=1))
                    return
