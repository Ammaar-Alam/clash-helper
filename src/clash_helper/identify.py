"""Name enemy deploys with the open Clash Royale unit detector.

The weights come from ClashRoyaleBuildABot and are AGPL-3.0, so they are
downloaded into the user cache on first use instead of living in this repo.
"""

from __future__ import annotations

import ast
import urllib.request
from collections import Counter
from functools import cache
from pathlib import Path

import cv2
import numpy as np

from clash_helper.cards import all_cards, card

MODEL_URL = "https://raw.githubusercontent.com/Pbatch/ClashRoyaleBuildABot/main/clashroyalebuildabot/models/units_M_480x352.onnx"
MODEL_PATH = Path.home() / "Library/Caches/clash-helper/units_M_480x352.onnx"
INPUT_W, INPUT_H = 352, 480
CROP_TOP, CROP_BOTTOM = 150, 1200  # arena band of the canonical screen
MIN_CONF = 0.3


class Detector:
    def __init__(self) -> None:
        import onnxruntime as ort

        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            print("downloading unit detector (31 MB, once)", flush=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(MODEL_PATH), options, providers=["CPUExecutionProvider"])
        self.names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map["names"])

    def detect(self, frame: np.ndarray) -> list[tuple[str, float, int, int, int]]:
        """Units as name, confidence, center x, top y, bottom y in screen pixels."""
        crop = frame[CROP_TOP:CROP_BOTTOM]
        h, w = crop.shape[:2]
        scale = min(INPUT_W / w, INPUT_H / h)
        nw, nh = int(w * scale), int(h * scale)
        px, py = (INPUT_W - nw) // 2, (INPUT_H - nh) // 2
        canvas = np.full((INPUT_H, INPUT_W, 3), 114, np.uint8)
        canvas[py : py + nh, px : px + nw] = cv2.resize(crop, (nw, nh))
        batch = (canvas[:, :, ::-1].transpose(2, 0, 1)[None] / 255).astype(np.float16)
        rows = self.session.run(None, {"images": batch})[0][0]
        return [
            (
                self.names[int(k)],
                float(c),
                int(((x1 + x2) / 2 - px) / scale),
                int((y1 - py) / scale) + CROP_TOP,
                int((y2 - py) / scale) + CROP_TOP,
            )
            for x1, y1, x2, y2, c, k in rows
            if c >= MIN_CONF
        ]


@cache
def unit_cards() -> dict[str, list[str]]:
    """Detector unit to the cards that can put it on the arena."""
    mapping: dict[str, list[str]] = {}
    for c in all_cards().values():
        for unit in c.units:
            mapping.setdefault(unit, []).append(c.name)
    return mapping


def swarm_size(name: str) -> int:
    return {"Skeleton Army": 15, "Minion Horde": 6, "Goblin Gang": 6, "Royal Recruits": 6, "Barbarians": 5}.get(name, 3)


def name_deploy(votes: Counter, candidates: list[str] | None, units_seen: int) -> str | None:
    """Most likely card for one deploy, None when the detector saw nothing usable."""
    scores: Counter = Counter()
    for unit, weight in votes.items():
        for name in unit_cards().get(unit, []):
            if candidates is not None and name not in candidates:
                continue
            # the card whose main unit this is beats cards that only spawn it
            main = card(name).units[0] == unit
            scores[name] += weight * (1.0 if main else 0.6)
    if not scores:
        return None
    best = scores.most_common(3)
    # several cards share one unit, the unit count separates them
    shared = [n for n, s in best if s >= best[0][1] * 0.6]
    if len(shared) > 1:
        return min(shared, key=lambda n: abs(swarm_size(n) - units_seen) if card(n).has("swarm") else abs(1 - units_seen))
    return best[0][0]
