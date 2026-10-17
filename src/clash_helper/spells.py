"""Enemy spells, which land without a deploy timer.

Each detector only runs when the opponent's deck holds that spell, which keeps
look-alikes such as the player's own spells and arena effects from tripping it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from clash_helper import vision

SAME_SPELL_SECONDS = 2.0
BARREL_SPAWNS = {"Goblin Barrel": 3, "Skeleton Barrel": 5}


@dataclass
class Cast:
    card: str
    time: float
    x: int
    y: int


def _blobs(mask: np.ndarray, min_area: int, max_area: int, fill: float = 0.5) -> list[tuple[int, int]]:
    # textures and shading split a spell into pieces, close them back up
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    found = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if min_area <= area <= max_area and 0.6 <= w / h <= 1.6 and area >= fill * w * h:
            found.append((int(centers[i][0]), int(centers[i][1]) + vision.ARENA_TOP))
    return found


@dataclass
class SpellWatcher:
    deck: list[str] = field(default_factory=list)
    balls: list[tuple[int, int]] = field(default_factory=list)  # snowball candidates last frame
    glows: list[tuple[int, int]] = field(default_factory=list)  # vines candidates last frame
    recent: list[Cast] = field(default_factory=list)
    badge_history: list[tuple[float, list[tuple[int, int]]]] = field(default_factory=list)

    def _cast(self, name: str, now: float, x: int, y: int) -> Cast | None:
        if any(c.card == name and now - c.time < SAME_SPELL_SECONDS for c in self.recent):
            return None
        cast = Cast(name, now, x, y)
        self.recent = [c for c in self.recent if now - c.time < 10] + [cast]
        return cast

    def update(self, frame: np.ndarray, now: float, clocks: list[tuple[int, int, float]]) -> list[Cast]:
        casts: list[Cast] = []
        hsv = cv2.cvtColor(frame[vision.ARENA_TOP : vision.ARENA_BOTTOM], cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        if "Giant Snowball" in self.deck:
            # a big pale ball that moves down the arena from their side
            balls = _blobs((s < 60) & (v > 195), 2500, 12000)
            for x, y in balls:
                if any(15 < y - py < 160 and abs(x - px) < 120 for px, py in self.balls):
                    if cast := self._cast("Giant Snowball", now, x, y):
                        casts.append(cast)
            self.balls = balls

        if "Vines" in self.deck:
            # a lime flash that was not there a frame ago, arena grass is the same color but never moves
            glows = _blobs((h >= 33) & (h <= 60) & (s > 110) & (v > 175), 1500, 20000, fill=0.3)
            # plants line the arena edges on some maps
            glows = [(x, y) for x, y in glows if 70 < x < 594]
            for x, y in glows:
                if not any(abs(x - gx) + abs(y - gy) < 60 for gx, gy in self.glows):
                    if cast := self._cast("Vines", now, x, y):
                        casts.append(cast)
            self.glows = glows

        barrels = [n for n in BARREL_SPAWNS if n in self.deck]
        if barrels:
            casts += self._barrels(frame, now, clocks, barrels)
        return casts

    def _barrels(self, frame: np.ndarray, now: float, clocks: list[tuple[int, int, float]], barrels: list[str]) -> list[Cast]:
        """Several enemy units popping up by your towers with no timer came out of a barrel."""
        badges = [(x, y) for x, y, _ in vision.level_badges(frame, enemy=True) if y > vision.RIVER_Y + 100]
        before = [b for t, bs in self.badge_history if now - t < 1.5 for b in bs]
        self.badge_history = [(t, bs) for t, bs in self.badge_history if now - t < 1.5] + [(now, badges)]
        fresh = [(x, y) for x, y in badges if not any(abs(x - bx) + abs(y - by) < 90 for bx, by in before)]
        if len(fresh) < 2:
            return []
        x, y = int(np.mean([f[0] for f in fresh])), int(np.mean([f[1] for f in fresh]))
        if any(abs(x - cx) + abs(y - cy) < 150 and now - t < 2 for cx, cy, t in clocks):
            return []
        cast = self._cast(barrels[0], now, x, y)
        return [cast] if cast else []
