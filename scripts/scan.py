"""Cache what the unit detector sees in every recording, so experiments replay without the GPU.

    python scripts/scan.py              # every recording in captures/dataset.json not scanned yet
    python scripts/scan.py match4       # one recording

Writes captures/scans/<id>.json.gz with one entry per analysed frame:
[time, [[name, card, enemy, conf, x, top, bottom, track], ...]] in canonical screen pixels.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

from clash_helper.cli import video_frames
from clash_helper.kcr import Detector

CAPTURES = Path(__file__).resolve().parents[1] / "captures"
FPS = 6.0


def scan(item: dict, detector: Detector) -> None:
    out = CAPTURES / "scans" / f"{item['id']}.json.gz"
    if out.exists():
        return
    out.parent.mkdir(exist_ok=True)
    rows = []
    started = time.time()
    for frame, now in video_frames(Path(item["video"]), FPS):
        detector.submit(frame, now)
        while detector.busy:
            time.sleep(0.002)
        _, units = detector.result()
        rows.append([round(now, 3), [[u.name, u.card, int(u.enemy), round(u.conf, 3), u.x, u.top, u.bottom, u.track]
                                     for u in units]])
    with gzip.open(out, "wt") as f:
        json.dump(rows, f)
    print(f"{item['id']}: {len(rows)} frames in {time.time() - started:.0f}s", flush=True)


def main() -> None:
    items = json.loads((CAPTURES / "dataset.json").read_text())
    wanted = set(sys.argv[1:])
    detector = Detector()
    for item in items:
        if not wanted or item["id"] in wanted:
            scan(item, detector)
    detector.close()


if __name__ == "__main__":
    main()
