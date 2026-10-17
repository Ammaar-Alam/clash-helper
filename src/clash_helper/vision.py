"""Read the canonical 664x1440 screen from screen.normalize."""

from __future__ import annotations

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


