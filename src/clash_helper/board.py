"""What is on the arena right now: tower health and whose units are where."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from clash_helper import vision

LANES = ("left", "right")
MID_X = 332
# tower health numbers without the crown icon, as y0, y1, x0, x1
TOWER_HP = {
    ("their", "left"): (250, 280, 108, 198),
    ("their", "right"): (250, 280, 496, 588),
    ("my", "left"): (900, 932, 108, 198),
    ("my", "right"): (900, 932, 496, 588),
}
TOWER_EVERY = 2.0  # seconds between health reads, text recognition is the slow part
GONE_AFTER = 3  # blank reads in a row before a damaged tower counts as destroyed
HOLD_FRAMES = 6  # badges flicker in fights, remember crossings for about a second


def lane(x: int) -> str:
    return "left" if x < MID_X else "right"


@dataclass
class Board:
    their_towers: dict[str, int | None] = field(default_factory=lambda: dict.fromkeys(LANES))
    my_towers: dict[str, int | None] = field(default_factory=lambda: dict.fromkeys(LANES))
    # enemy units that crossed onto the player's half, the danger
    enemy_home: dict[str, int] = field(default_factory=lambda: dict.fromkeys(LANES, 0))
    # the player's units past the river, a push in progress
    mine_away: dict[str, int] = field(default_factory=lambda: dict.fromkeys(LANES, 0))
    # the player's units still on their own half, defenders or a push forming
    mine_home: dict[str, int] = field(default_factory=lambda: dict.fromkeys(LANES, 0))
    my_units: list[tuple[str, int, int]] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "their_towers": self.their_towers,
            "my_towers": self.my_towers,
            "enemy_home": self.enemy_home,
            "mine_away": self.mine_away,
            "mine_home": self.mine_home,
        }


def _number(lines: list[str]) -> int | None:
    digits = re.findall(r"\d{2,4}", " ".join(lines).replace("O", "0").replace("o", "0"))
    return int(digits[-1]) if digits else None


class BoardReader:
    def __init__(self) -> None:
        self.board = Board()
        self.last_read = -TOWER_EVERY
        self.blanks: dict[tuple[str, str], int] = {}
        self.crossed: deque[dict[str, int]] = deque(maxlen=HOLD_FRAMES)

    def towers(self, frame: np.ndarray, now: float) -> None:
        if now - self.last_read < TOWER_EVERY:
            return
        self.last_read = now
        for (side, which), (y0, y1, x0, x1) in TOWER_HP.items():
            crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            hp = _number(vision.read_text(crop))
            towers = self.board.their_towers if side == "their" else self.board.my_towers
            known = towers[which]
            if hp is not None and (known is None or hp <= known):
                # health only ever goes down, a bigger number is a misread
                towers[which] = hp
                self.blanks[(side, which)] = 0
            elif hp is None and known:
                self.blanks[(side, which)] = self.blanks.get((side, which), 0) + 1
                if self.blanks[(side, which)] >= GONE_AFTER:
                    towers[which] = 0

    def units(self, frame: np.ndarray, detections: list[tuple[str, float, int, int, int]] | None) -> None:
        enemy = vision.level_badges(frame, enemy=True)
        self.crossed.append({ln: sum(1 for x, y, _ in enemy if lane(x) == ln and y > vision.RIVER_Y) for ln in LANES})
        self.board.enemy_home = {ln: max(c[ln] for c in self.crossed) for ln in LANES}
        if detections is None:
            return
        mine = [(n, x, bottom) for n, _, x, top, bottom in detections if vision.unit_side(frame, x, top) == "mine"]
        self.board.my_units = mine
        self.board.mine_away = {ln: sum(1 for _, x, y in mine if lane(x) == ln and y < vision.RIVER_Y) for ln in LANES}
        self.board.mine_home = {ln: sum(1 for _, x, y in mine if lane(x) == ln and y >= vision.RIVER_Y) for ln in LANES}
