"""Read the canonical 664x1440 screen from screen.normalize."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

# arena rows from just above the enemy king tower down to the hand panel,
# cards dropped behind their king tower land as high as about 170
ARENA_TOP, ARENA_BOTTOM = 140, 1180
RIVER_Y = 640
# elixir bar, filled segments are solid magenta
BAR_Y0, BAR_Y1 = 1372, 1388
BAR_EMPTY_X, BAR_FULL_X = 180, 640
# double elixir drop under the timer
BADGE_X2 = (slice(160, 240), slice(580, 664))
# hand cards then the small next card, as y0, y1, x0, x1
HAND_SLOTS = [(1210, 1355, 152 + i * 124, 268 + i * 124) for i in range(4)]
NEXT_SLOT = (1325, 1395, 35, 90)


def _hsv(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def _magenta(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h > 140) & (h < 168) & (s > 70) & (v > 150)


def own_elixir(frame: np.ndarray) -> int:
    filled = _magenta(_hsv(frame[BAR_Y0:BAR_Y1])).mean(axis=0) > 0.5
    cols = np.where(filled[BAR_EMPTY_X:])[0]
    end = BAR_EMPTY_X + (int(cols.max()) if len(cols) else 0)
    step = (BAR_FULL_X - BAR_EMPTY_X) / 10
    return int(np.clip(round((end - BAR_EMPTY_X) / step), 0, 10))


def elixir_multiplier(frame: np.ndarray) -> float:
    # ponytail: x2 only, add an x2 vs x3 template once an overtime recording shows the x3 badge
    return 2.0 if _magenta(_hsv(frame[BADGE_X2])).mean() > 0.08 else 1.0


NEXT_LABEL = cv2.imread(str(Path(__file__).parent / "data" / "next_label.png"), cv2.IMREAD_GRAYSCALE)


def in_match(frame: np.ndarray) -> bool:
    """The "Next:" label beside the hand only exists during a match, on every arena."""
    # a wide search area, recordings shift the screen by a few pixels now and then
    area = cv2.cvtColor(frame[1200:1420, 0:170], cv2.COLOR_BGR2GRAY)
    # matches score 0.88 and up, menus and end screens 0.46 at most
    return bool(cv2.matchTemplate(area, NEXT_LABEL, cv2.TM_CCOEFF_NORMED).max() > 0.7)


def read_text(image: np.ndarray) -> list[str]:
    """Lines of text top to bottom via the macOS text recognizer."""
    import Vision
    from Foundation import NSData

    ok, png = cv2.imencode(".png", image)
    if not ok:
        return []
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setUsesLanguageCorrection_(False)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
        NSData.dataWithBytes_length_(png.tobytes(), len(png)), None
    )
    handler.performRequests_error_([request], None)
    results = request.results() or []
    # Vision boxes are bottom-left origin, so higher y means higher on screen
    ordered = sorted(results, key=lambda r: -r.boundingBox().origin.y)
    return [r.topCandidates_(1)[0].string() for r in ordered]


TIMER = cv2.imread(str(Path(__file__).parent / "data" / "deploy_timer.png"))
TIMER_MATCH = 0.32  # real timers score 0.4 and up, red floor splatter about 0.2


def time_left(frame: np.ndarray) -> int | None:
    """Seconds on the match clock in the top right, None when it cannot be read."""
    for line in read_text(frame[95:170, 550:664]):
        found = re.search(r"(\d):([0-5]\d)", line.replace(" ", ""))
        if found:
            return int(found.group(1)) * 60 + int(found.group(2))
    return None


SKY = slice(40, 135)  # the score strip at the top, a bolt from the sky drops straight through it
# parts of the strip that change on their own or never show anything, x0, x1, y0, y1: the shimmering
# name, the match clock, the screen edges that pulse red in overtime and the camera notch
SKY_RESTLESS = [(0, 250, 100, 135), (525, 664, 90, 135), (538, 664, 40, 135), (0, 72, 40, 135), (234, 440, 36, 100)]
SKY_ROWS = 40  # a column is judged on its open rows and needs this many, under the notch only the king tower shows
BOLT_WIDE = 8  # windows near the brightest one that may light up too, an emote or a screen shift lights many
_SKY_OPEN = np.ones((SKY.stop - SKY.start, 664), bool)
for _x0, _x1, _y0, _y1 in SKY_RESTLESS:
    _SKY_OPEN[max(0, _y0 - SKY.start) : _y1 - SKY.start, _x0:_x1] = False
_RING = np.zeros((2 * 92 + 41, 2 * 120 + 41), np.float32)
cv2.ellipse(_RING, (140, 112), (120, 92), 0, 0, 360, 1.0, 9)
_RING /= _RING.sum()


def sky(frame: np.ndarray) -> np.ndarray:
    """Brightness of the top strip, compare two frames with sky_bolt."""
    return _hsv(frame[SKY])[..., 2].astype(np.int16)


def _bands(rows: np.ndarray) -> np.ndarray:
    """Per row, whether anything in each 12 px window is set, a bolt wanders a few pixels."""
    width = rows.shape[1] // 4 * 4
    cols = rows[:, :width].reshape(rows.shape[0], -1, 4).any(axis=2)
    return np.stack([cols[:, i : i + cols.shape[1] - 2] for i in range(3)]).any(axis=0)


_SKY_OPEN_ROWS = _bands(_SKY_OPEN).sum(axis=0)


def sky_bolt(now_sky: np.ndarray, before: np.ndarray) -> tuple[float, int]:
    """How much of a thin column brightened through the open part of the top strip, and where.
    A Zap and an Electro Wizard both drop in as a bolt from the sky."""
    risen = _bands(((now_sky - before) > 25) & _SKY_OPEN).sum(axis=0)
    share = np.where(_SKY_OPEN_ROWS >= SKY_ROWS, risen / np.maximum(_SKY_OPEN_ROWS, 1), 0.0)
    best = int(np.argmax(share))
    # a bolt is thin, the windows 40 px around it stay mostly dark
    around = share[max(0, best - 10) : best + 11]
    if (around >= share[best] / 2).sum() > BOLT_WIDE:
        return 0.0, best * 4 + 6
    return round(float(share[best]), 3), best * 4 + 6


def arrows_ring(frame: np.ndarray) -> tuple[float, int, int]:
    """Strength and center of the salmon impact ring enemy Arrows leave on the ground."""
    hsv = _hsv(frame[ARENA_TOP:ARENA_BOTTOM])
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h >= 2) & (h <= 11) & (s >= 95) & (s <= 160) & (v >= 190)).astype(np.float32)
    score = cv2.filter2D(mask, -1, _RING, borderType=cv2.BORDER_CONSTANT)
    _, best, _, (x, y) = cv2.minMaxLoc(score)
    return round(float(best), 3), int(x), int(y) + ARENA_TOP


def deploy_clocks(frame: np.ndarray) -> list[tuple[int, int]]:
    """Centers of the red stopwatch shown where an enemy card is landing."""
    hsv = _hsv(frame[ARENA_TOP:ARENA_BOTTOM])
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h <= 6) | (h >= 172)) & (s > 140) & (v > 170)).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(red)
    found = []
    for i in range(1, count):
        x, y, w, hgt, area = stats[i]
        # a filled disc, the needle splits nothing because it sits inside
        if not (22 <= w <= 48 and 22 <= hgt <= 48 and 0.8 <= w / hgt <= 1.25):
            continue
        if not (0.55 <= area / (w * hgt) <= 0.9):
            continue
        cx, cy = int(centers[i][0]), int(centers[i][1]) + ARENA_TOP
        # a red disc is common on some arenas, the gold rim, knob and needle are not
        around = frame[max(0, cy - 40) : cy + 40, max(0, cx - 38) : cx + 38]
        if around.shape[0] < TIMER.shape[0] or around.shape[1] < TIMER.shape[1]:
            continue
        if cv2.matchTemplate(around, TIMER, cv2.TM_CCOEFF_NORMED).max() >= TIMER_MATCH:
            found.append((cx, cy))
    return found


