"""Read the canonical 664x1440 screen from screen.normalize."""

from __future__ import annotations

import cv2
import numpy as np

# arena rows between the enemy king tower top and the hand panel
ARENA_TOP, ARENA_BOTTOM = 190, 1180
RIVER_Y = 640
# elixir bar, filled segments are solid magenta
BAR_Y0, BAR_Y1 = 1372, 1388
BAR_EMPTY_X, BAR_FULL_X = 180, 640
# double elixir drop under the timer
BADGE_X2 = (slice(160, 240), slice(580, 664))
# hand cards then the small next card, as y0, y1, x0, x1
HAND_SLOTS = [(1212, 1340, 150 + i * 124, 265 + i * 124) for i in range(4)]
NEXT_SLOT = (1325, 1395, 35, 90)


def _hsv(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def _magenta(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h > 140) & (h < 165) & (s > 120) & (v > 150)


def own_elixir(frame: np.ndarray) -> int:
    filled = _magenta(_hsv(frame[BAR_Y0:BAR_Y1])).mean(axis=0) > 0.5
    cols = np.where(filled[BAR_EMPTY_X:])[0]
    end = BAR_EMPTY_X + (int(cols.max()) if len(cols) else 0)
    step = (BAR_FULL_X - BAR_EMPTY_X) / 10
    return int(np.clip(round((end - BAR_EMPTY_X) / step), 0, 10))


def elixir_multiplier(frame: np.ndarray) -> float:
    # ponytail: x2 only, add an x2 vs x3 template once an overtime recording shows the x3 badge
    return 2.0 if _magenta(_hsv(frame[BADGE_X2])).mean() > 0.08 else 1.0


def in_match(frame: np.ndarray) -> bool:
    """The blue hand panel and the elixir bar frame only exist during a match."""
    bar = frame[BAR_Y0:BAR_Y1, 200:640]
    # every bar pixel is either empty dark blue, filling light blue or full magenta
    blue = (bar[..., 0] > 70) & (bar[..., 2] < 90)
    filled = _magenta(_hsv(bar))
    return bool((blue | filled).mean() > 0.85)


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
        found.append((int(centers[i][0]), int(centers[i][1]) + ARENA_TOP))
    return found


def level_badges(frame: np.ndarray, enemy: bool) -> list[tuple[int, int]]:
    """Centers of unit level badges, crimson for the enemy and blue for the player."""
    hsv = _hsv(frame[ARENA_TOP:ARENA_BOTTOM])
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if enemy:
        mask = ((h >= 165) | (h <= 3)) & (s > 150) & (v > 110)
    else:
        mask = (h >= 100) & (h <= 115) & (s > 170) & (v > 150)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # the level digit is white or gold, tower trim of the same red has none
    digit = (v > 200) & ((s < 90) | ((h >= 18) & (h <= 35)))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    found = []
    for i in range(1, count):
        x, y, w, hgt, area = stats[i]
        # badges are small near-square blocks, health bars are long and thin
        if not (12 <= w <= 34 and 14 <= hgt <= 34 and 0.6 <= w / hgt <= 1.6 and area >= 0.3 * w * hgt):
            continue
        if digit[y : y + hgt, x : x + w].mean() < 0.12:
            continue
        found.append((int(centers[i][0]), int(centers[i][1]) + ARENA_TOP))
    return found
