"""Cut image crops around every labeled play, the training data for naming plays.

    python research/crops.py            # every labeled recording
    python research/crops.py match1     # some

Writes captures/crops/<id>.npz: crops (N, K, 224, 224, 3) uint8 BGR, cards, times, xs, ys.
Troops are cut where they stand in the first second, spells along their flight into the impact.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import truth_plays  # noqa: E402

from clash_helper.cards import card  # noqa: E402
from clash_helper.cli import video_frames  # noqa: E402

CAPTURES = Path(__file__).resolve().parents[1] / "captures"
TROOP_OFFSETS = (0.3, 0.7, 1.1)  # seconds after landing
SPELL_OFFSETS = (-0.5, -0.2, 0.1)  # seconds around impact
SIZE = 224


def crop(frame: np.ndarray, x: int, y: int, half: int) -> np.ndarray:
    padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_CONSTANT)
    c = padded[y : y + 2 * half, x : x + 2 * half]
    return cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def scan_for(item: dict) -> list | None:
    path = CAPTURES / "scans" / f"{item['id']}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt") as f:
        return json.load(f)


def unit_box(scan: list, now: float, x: int, y: int, spell: bool) -> tuple[int, int, int] | None:
    """The detector's own box around the enemy unit or spell nearest the play, as center x, center y, side."""
    times = [row[0] for row in scan]
    i = min(range(len(times)), key=lambda j: abs(times[j] - now))
    if abs(times[i] - now) > 0.2:
        return None
    best = None
    for name, c, enemy, conf, ux, top, bottom, track in scan[i][1]:
        if not c or not enemy or card(c).spell != spell:
            continue
        cy = (top + bottom) / 2
        d = abs(ux - x) + abs(bottom - y if not spell else cy - y)
        if d < (160 if spell else 110) and (best is None or d - 60 * conf < best[0]):
            best = (d - 60 * conf, ux, cy, max(48, bottom - top))
    if best is None:
        return None
    _, ux, cy, h = best
    return int(ux), int(cy), int(h * 0.65)


def cut(item: dict) -> None:
    plays = [p for p in truth_plays(json.loads(Path(item["labels"]).read_text())) if p.get("x", -1) >= 0]
    scan = scan_for(item)
    wanted: dict[float, list[tuple[int, int]]] = {}
    for i, p in enumerate(plays):
        offsets = SPELL_OFFSETS if card(p["card"]).spell else TROOP_OFFSETS
        for k, d in enumerate(offsets):
            wanted.setdefault(round(p["time"] + d, 2), []).append((i, k))
    out = np.zeros((len(plays), 3, SIZE, SIZE, 3), np.uint8)
    times = sorted(wanted)
    j = 0
    for frame, now in video_frames(Path(item["video"]), 6):
        while j < len(times) and times[j] <= now + 0.09:
            for i, k in wanted[times[j]]:
                p = plays[i]
                spell = card(p["card"]).spell
                box = unit_box(scan, now, int(p["x"]), int(p["y"]), spell) if scan else None
                if box:
                    # tight on the unit itself, the arena around it says nothing about the card
                    out[i, k] = crop(frame, box[0], box[1], box[2])
                else:
                    out[i, k] = crop(frame, int(p["x"]), int(p["y"]) - (0 if spell else 40), 90 if spell else 60)
            j += 1
        if j >= len(times):
            break
    np.savez_compressed(CAPTURES / "crops" / f"{item['id']}.npz", crops=out, cards=[p["card"] for p in plays],
                        times=[p["time"] for p in plays], xs=[p["x"] for p in plays], ys=[p["y"] for p in plays])
    print(f"{item['id']}: {len(plays)} plays", flush=True)


def main() -> None:
    (CAPTURES / "crops").mkdir(exist_ok=True)
    items = [i for i in json.loads((CAPTURES / "dataset.json").read_text()) if i["labels"]]
    for item in items:
        if not sys.argv[1:] or item["id"] in sys.argv[1:]:
            cut(item)


if __name__ == "__main__":
    main()
