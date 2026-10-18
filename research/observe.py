"""Cache everything each frame shows, so experiments replay a match in seconds.

    python research/observe.py            # every recording with a scan
    python research/observe.py match4     # some

Needs the detector scan from scripts/scan.py. Writes captures/obs/<id>.json.gz, one
Observation per frame at 6 fps, with tower health read once a second.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from clash_helper import vision
from clash_helper.board import read_towers
from clash_helper.cli import video_frames
from clash_helper.game import Observation
from clash_helper.hand import HandReader

CAPTURES = Path(__file__).resolve().parents[1] / "captures"
DATA = Path(__file__).resolve().parents[1] / "src/clash_helper/data/cards.json"


def hand_reader(names: list[str]) -> HandReader | None:
    icons = {c["name"]: c for c in json.loads(DATA.read_text())}
    deck = [{"name": n, "iconUrls": {"medium": icons[n]["icon"], "evolutionMedium": icons[n].get("evo_icon")},
             "evolutionLevel": 1 if icons[n].get("evo_icon") else 0} for n in names if n in icons and icons[n].get("icon")]
    return HandReader(deck) if len(deck) == 8 else None


def build(item: dict) -> None:
    out = CAPTURES / "obs" / f"{item['id']}.json.gz"
    if out.exists():
        return
    with gzip.open(CAPTURES / "scans" / f"{item['id']}.json.gz", "rt") as f:
        scan = json.load(f)
    reader = hand_reader(item.get("my_deck", []))
    rows, i, last_towers, clock_reads, last_sky = [], 0, -9.0, 0, None
    started = time.time()
    for frame, now in video_frames(Path(item["video"]), 6):
        obs = Observation(round(now, 3), vision.in_match(frame))
        if obs.in_match:
            if clock_reads < 8:
                # the first frames of a match say how far in the recording starts
                obs.time_left = vision.time_left(frame)
                clock_reads += 1
            obs.multiplier = vision.elixir_multiplier(frame)
            obs.elixir = vision.own_elixir(frame)
            obs.hand = reader.read(frame)[0] if reader else None
            obs.clocks = [list(c) for c in vision.deploy_clocks(frame)]
            sky = vision.sky(frame)
            if last_sky is not None:
                obs.bolt = vision.sky_bolt(sky, last_sky)
            last_sky = sky
            obs.ring = vision.arrows_ring(frame)
            while i + 1 < len(scan) and scan[i + 1][0] <= now + 1e-6:
                i += 1
            obs.units_at, obs.units = scan[i][0], scan[i][1]
            if now - last_towers >= 1.0:
                last_towers = now
                obs.towers = [[side, which, hp] for (side, which), hp in read_towers(frame).items()]
        rows.append(asdict(obs))
    out.parent.mkdir(exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump(rows, f)
    print(f"{item['id']}: {len(rows)} frames in {time.time() - started:.0f}s", flush=True)


def load(path: Path) -> list[Observation]:
    """Cached observations with detector units turned back into kcr.Unit."""
    from clash_helper import kcr

    with gzip.open(path, "rt") as f:
        rows = json.load(f)
    out = []
    for r in rows:
        r["units"] = [kcr.Unit(n, c, bool(e), conf, x, top, bottom, track) for n, c, e, conf, x, top, bottom, track in r["units"]]
        r["clocks"] = [tuple(c) for c in r["clocks"]]
        r["bolt"], r["ring"] = tuple(r.get("bolt", (0.0, 0))), tuple(r.get("ring", (0.0, 0, 0)))
        r["towers"] = {(side, which): hp for side, which, hp in r["towers"]} if r["towers"] is not None else None
        out.append(Observation(**r))
    return out


def main() -> None:
    items = [i for i in json.loads((CAPTURES / "dataset.json").read_text())
             if (CAPTURES / "scans" / f"{i['id']}.json.gz").exists()]
    for item in items:
        if not sys.argv[1:] or item["id"] in sys.argv[1:]:
            build(item)


if __name__ == "__main__":
    main()
