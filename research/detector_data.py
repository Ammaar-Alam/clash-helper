"""Training frames for fine tuning the unit detector on the labeled recordings.

    python research/play_tracks.py              # which tracks each labeled play put down, first
    python research/detector_data.py            # every labeled recording
    python research/detector_data.py match4     # some

Writes captures/detector/<id>/: arena crops (568x896, what the detector sees) at 2 fps and at 6 fps in
the 4 s after each labeled play, list.txt, and fixes.json. fixes.json says, per frame, which of the scan's
units belong to a labeled play: their class, where they stand, the class the true card gives them (null
to keep the detector's) and that they are theirs. research/colab_train.py runs the detector on the crops
on a GPU, keeps its boxes as labels and applies the fixes to the boxes that match those units.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import cv2

from clash_helper import kcr
from clash_helper.cards import card
from clash_helper.cli import video_frames
from clash_helper.game import SPAWNS

CAPTURES = Path(__file__).resolve().parents[1] / "captures"
OUT = CAPTURES / "detector"
FPS = 6.0
PLAY_SECONDS = (-0.3, 4.0)  # frames around a labeled play taken at the full rate
# cards whose units are another card's, a box of one of those units is that unit
PARTS = {"Goblin Gang": {"Goblins", "Spear Goblins"}, "Skeleton Army": {"Skeletons"}, "Three Musketeers": {"Musketeer"},
         "Minion Horde": {"Minions"}}


def unit_list() -> list[str]:
    """KataCR's global class names, index is the class number its labels use."""
    text = (kcr.ROOT / "KataCR/katacr/constants/label_list.py").read_text()
    start = text.index("unit_list = [") + len("unit_list = ")
    return eval(text[start : text.index("]", start) + 1])  # noqa: S307, a plain list of strings in a file we installed


def base_class(name: str, classes: list[str]) -> str | None:
    """The detector class for a card's unit, the plain one rather than an evolution."""
    plain = re.sub(r"[^a-z]", "", name.lower())
    found = [c for c in classes if kcr.card_of(c) == name and "evolution" not in c]
    return min(found, key=lambda c: abs(len(re.sub(r"[^a-z]", "", c)) - len(plain))) if found else None


def true_class(play: str, detected: str, classes: list[str]) -> str | None:
    """The class a labeled play's box should carry, None to keep the detector's own."""
    seen = kcr.card_of(detected)
    if seen == play or seen in PARTS.get(play, ()):
        return None
    if play in SPAWNS.get(seen, ()) or card(seen).spell != card(play).spell:
        return None  # a unit the card spawns, or a unit voting on a spell and the other way round
    if play in PARTS:
        return None  # which part this unit is cannot be told
    return base_class(play, classes)


def build(item: dict, plays: list[dict], classes: list[str]) -> tuple[int, int, int]:
    with gzip.open(CAPTURES / "scans" / f"{item['id']}.json.gz", "rt") as f:
        scan = {t: units for t, units in json.load(f)}
    folder = OUT / item["id"]
    folder.mkdir(parents=True, exist_ok=True)
    x0, y0, w, h = kcr.ARENA
    names, fixes, renamed = [], {}, 0
    for i, (frame, now) in enumerate(video_frames(Path(item["video"]), FPS)):
        t = round(now, 3)
        live = [p for p in plays if PLAY_SECONDS[0] <= now - p["time"] <= PLAY_SECONDS[1]]
        if t not in scan or not (i % 3 == 0 or live):
            continue
        stem = f"{int(round(now * 1000)):07d}"
        crop = cv2.resize(frame[y0 : y0 + h, x0 : x0 + w], kcr.INPUT, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(folder / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        names.append(f"{item['id']}/{stem}.jpg")
        here = []
        for name, _, _, _, x, _, bottom, track in scan[t]:
            play = next((p for p in live if track in p["tracks"]), None)
            if play is not None and name in classes:
                fixed = true_class(play["card"], name, classes)
                renamed += fixed is not None
                here.append([name, x, bottom, fixed, 1])
        if here:
            fixes[stem] = here
    (folder / "list.txt").write_text("\n".join(names))
    (folder / "fixes.json").write_text(json.dumps(fixes))
    return len(names), sum(map(len, fixes.values())), renamed


def main() -> None:
    plays = json.loads((CAPTURES / "play_tracks.json").read_text())
    classes = unit_list()
    items = [i for i in json.loads((CAPTURES / "dataset.json").read_text())
             if i["labels"] and (not sys.argv[1:] or i["id"] in sys.argv[1:])]
    for item in items:
        frames, marked, renamed = build(item, plays.get(item["id"], []), classes)
        print(f"{item['id']:<28} {frames} frames, {marked} play units marked, {renamed} renamed", flush=True)


if __name__ == "__main__":
    main()
