"""What is on the arena right now: tower health and whose units are where."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from clash_helper import vision

LANES = ("left", "right")
MID_X = 332
# tower health numbers without the crown icon, as y0, y1, x0, x1. The rows are a fallback,
# tower troops differ in height so the detector's health bar box gives them
TOWER_HP = {
    ("their", "left"): (250, 280, 108, 198),
    ("their", "right"): (250, 280, 496, 588),
    ("my", "left"): (900, 932, 108, 198),
    ("my", "right"): (900, 932, 496, 588),
}
TOWER_EVERY = 1.0  # seconds between health reads, text recognition is the slow part
PRINCESS_TOWERS = {"queen-tower", "cannoneer-tower", "dagger-duchess-tower"}
HEALTH_BARS = {"tower-bar", "dagger-duchess-tower-bar"}
TOWER_GONE = 4.0  # seconds the detector may miss a princess tower before it counts as destroyed


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
    # three or four digits, the level badge beside the number reads as two
    digits = re.findall(r"\d{3,4}", " ".join(lines).replace("O", "0").replace("o", "0"))
    return int(digits[-1]) if digits else None


def read_towers(frame: np.ndarray, rows: dict | None = None) -> dict[tuple[str, str], int | None]:
    """The four princess tower health numbers as text recognition reads them, None when unreadable.
    `rows` holds the top and bottom of each health bar where the detector last saw it."""
    reads = {}
    for key, (y0, y1, x0, x1) in TOWER_HP.items():
        y0, y1 = (rows or {}).get(key, (y0, y1))
        crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        reads[key] = _number(vision.read_text(crop))
    return reads


class BoardReader:
    def __init__(self) -> None:
        self.board = Board()
        self.last_read = -TOWER_EVERY
        self.last_seen: dict[tuple[str, str], int | None] = {}
        self.rows: dict[tuple[str, str], tuple[int, int]] = {}  # each health bar's top and bottom
        self.standing: dict[tuple[str, str], float] = {}  # when the detector last saw each princess tower
        self.worker: threading.Thread | None = None

    def towers(self, frame: np.ndarray, now: float) -> None:
        """Read tower health off the main loop, text recognition takes a few hundred ms."""
        if now - self.last_read < TOWER_EVERY or (self.worker and self.worker.is_alive()):
            return
        self.last_read = now
        self.worker = threading.Thread(target=lambda f, r: self.apply(read_towers(f, r)),
                                       args=(frame.copy(), dict(self.rows)), daemon=True)
        self.worker.start()

    def apply(self, reads: dict[tuple[str, str], int | None]) -> None:
        """Fold one round of health reads into the board."""
        for (side, which), hp in reads.items():
            towers = self.board.their_towers if side == "their" else self.board.my_towers
            known = towers[which]
            seen = self.last_seen.get((side, which))
            self.last_seen[(side, which)] = hp
            # health only goes down, and a number counts once two reads in a row agree
            agreed = hp is not None and seen is not None and abs(hp - seen) <= max(20, 0.1 * seen)
            if agreed and known != 0 and (known is None or hp <= known):
                towers[which] = hp

    def see(self, units: list, at: float) -> None:
        """Where each health bar sits and which princess towers still stand, from the unit detector.
        A number hidden behind a unit or a card name says nothing, a tower gone from view does."""
        for u in units:
            key = ("their" if u.bottom < vision.RIVER_Y else "my", lane(u.x))
            if u.name in HEALTH_BARS:
                self.rows[key] = (u.top, u.bottom)
            elif u.name in PRINCESS_TOWERS:
                self.standing[key] = at
        # ruins now and then pass for a tower for a moment, a destroyed tower stays destroyed
        for (side, which), last in self.standing.items():
            if at - last > TOWER_GONE:
                (self.board.their_towers if side == "their" else self.board.my_towers)[which] = 0


# arena tiles, 18 columns and 32 rows counted from the player's back edge,
# fitted so tower sprites sit on their tiles in the canonical screen
TILE_X0, TILE_W = 17.5, 35.0
TOWER_ROW, TOWER_Y, TILE_H = 6.5, 900.0, 29.47
LANE_COL = {"left": 3, "right": 14}
RIVER_ROW = 15


def tile(col: float, row: float) -> tuple[int, int]:
    """Center of a tile in screen pixels, fractional tiles allowed."""
    return round(TILE_X0 + (col + 0.5) * TILE_W), round(TOWER_Y - (row - TOWER_ROW) * TILE_H)


def row_of(y: float) -> float:
    return TOWER_ROW + (TOWER_Y - y) / TILE_H
