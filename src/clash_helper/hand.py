"""Recognize the player's own hand from card art in the API."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from clash_helper import vision

ICONS = Path.home() / "Library/Caches/clash-helper/icons"
SIZE = (60, 72)  # art only, without the frame and the cost bubble
MIN_SCORE = 0.45


def _art(image: np.ndarray) -> np.ndarray:
    """Grayscale middle of a card, so unaffordable grey cards still match."""
    h, w = image.shape[:2]
    middle = image[int(h * 0.12) : int(h * 0.72), int(w * 0.14) : int(w * 0.86)]
    gray = cv2.cvtColor(middle, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(cv2.resize(gray, SIZE, interpolation=cv2.INTER_AREA))


def _icon(url: str) -> np.ndarray:
    # cards/300/x.png and cardevolutions/300/x.png share a file name
    path = ICONS / url.split("/")[-3] / url.split("/")[-1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba.shape[2] == 3:
        return rgba
    # the art sits in a transparent canvas, keep only the card itself
    ys, xs = np.where(rgba[:, :, 3] > 128)
    rgba = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    alpha = rgba[:, :, 3:] / 255
    return (rgba[:, :, :3] * alpha + 40 * (1 - alpha)).astype(np.uint8)


class HandReader:
    def __init__(self, deck: list[dict]) -> None:
        """`deck` is the API currentDeck, evolved cards also get their evolved art."""
        self.refs: list[tuple[str, np.ndarray]] = []
        for c in deck:
            urls = [c["iconUrls"]["medium"]]
            if c.get("evolutionLevel") and c["iconUrls"].get("evolutionMedium"):
                urls.append(c["iconUrls"]["evolutionMedium"])
            for url in urls:
                self.refs.append((c["name"], _art(_icon(url))))

    def _match(self, crop: np.ndarray) -> list[tuple[float, str]]:
        art = _art(crop)
        scores: dict[str, float] = {}
        for name, ref in self.refs:
            score = float(cv2.matchTemplate(art, ref, cv2.TM_CCOEFF_NORMED)[0, 0])
            scores[name] = max(scores.get(name, -1.0), score)
        return sorted(((s, n) for n, s in scores.items()), reverse=True)

    def read(self, frame: np.ndarray) -> tuple[list[str | None], str | None]:
        """The 4 hand cards left to right and the next card, None where unsure."""
        crops = [frame[y0:y1, x0:x1] for y0, y1, x0, x1 in vision.HAND_SLOTS]
        y0, y1, x0, x1 = vision.NEXT_SLOT
        crops.append(frame[y0:y1, x0:x1])
        ranked = [self._match(c) for c in crops]
        # greedy unique assignment, the 5 visible cards are always different
        names: list[str | None] = [None] * 5
        taken: set[str] = set()
        pairs = sorted(((s, i, n) for i, r in enumerate(ranked) for s, n in r), reverse=True)
        for score, i, name in pairs:
            if score < MIN_SCORE:
                break
            if names[i] is None and name not in taken:
                names[i] = name
                taken.add(name)
        return names[:4], names[4]
