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
# tower health numbers without the crown icon, as y0, y1, x0, x1
TOWER_HP = {
    ("their", "left"): (250, 280, 108, 198),
    ("their", "right"): (250, 280, 496, 588),
    ("my", "left"): (900, 932, 108, 198),
    ("my", "right"): (900, 932, 496, 588),
}
TOWER_EVERY = 1.0  # seconds between health reads, text recognition is the slow part
GONE_AFTER = 3  # blank reads in a row before a damaged tower counts as destroyed


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


def read_towers(frame: np.ndarray) -> dict[tuple[str, str], int | None]:
    """The four princess tower health numbers as text recognition reads them, None when unreadable."""
    reads = {}
    for key, (y0, y1, x0, x1) in TOWER_HP.items():
        crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        reads[key] = _number(vision.read_text(crop))
    return reads


class BoardReader:
    def __init__(self) -> None:
        self.board = Board()
        self.last_read = -TOWER_EVERY
        self.blanks: dict[tuple[str, str], int] = {}
        self.last_seen: dict[tuple[str, str], int | None] = {}
        self.worker: threading.Thread | None = None

    def towers(self, frame: np.ndarray, now: float) -> None:
        """Read tower health off the main loop, text recognition takes a few hundred ms."""
        if now - self.last_read < TOWER_EVERY or (self.worker and self.worker.is_alive()):
            return
        self.last_read = now
        self.worker = threading.Thread(target=lambda f: self.apply(read_towers(f)), args=(frame.copy(),), daemon=True)
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
            if agreed and (known is None or hp <= known):
                towers[which] = hp
                self.blanks[(side, which)] = 0
            elif hp is None and known:
                self.blanks[(side, which)] = self.blanks.get((side, which), 0) + 1
                if self.blanks[(side, which)] >= GONE_AFTER:
                    towers[which] = 0


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
