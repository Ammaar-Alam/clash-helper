"""Every candidate play the pipeline produces on the labeled recordings, paired with the truth.

    python research/events.py

Writes captures/events.json, one row per candidate: recording, time, place, whether it came
from a spell sighting, the detector's votes, the unit count and the true card or "none".
This is the training data for what each detector class really means.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import CAPTURES, CachedDetector, observations, pair, truth_plays  # noqa: E402

import gzip  # noqa: E402

from clash_helper.game import Match  # noqa: E402


def candidates(item: dict) -> list:
    """Every candidate play, with their deck known as in the evaluation."""
    with gzip.open(CAPTURES / "scans" / f"{item['id']}.json.gz", "rt") as f:
        match = Match(detector=CachedDetector(json.load(f)), lookup=False, sync=True)
    events: list = []
    for obs in observations(item, match, False):
        match.update(obs)
        if match.started is not None and not match.deck_from_api:
            match.opp.preload(item["their_deck"])
            match.deck_from_api = True
        if match.events:
            events = list(match.events)
    return events


def labeled(item: dict, events: list) -> list[dict]:
    events = sorted(events, key=lambda pe: pe[0].time)
    truth = truth_plays(json.loads(Path(item["labels"]).read_text()))
    hits = pair(truth, [(p.time, p.x, p.y, e.spell) for p, e in events])
    return [{"recording": item["id"], "time": round(p.time, 2), "x": p.x, "y": p.y, "spell": e.spell,
             "votes": dict(e.votes), "units": e.units, "truth": truth[h]["card"] if h is not None else "none",
             "deck": item["their_deck"]} for (p, e), h in zip(events, hits)]


def main() -> None:
    items = [i for i in json.loads((CAPTURES / "dataset.json").read_text())
             if i["labels"] and (CAPTURES / "scans" / f"{i['id']}.json.gz").exists()]
    rows = []
    for item in items:
        got = labeled(item, candidates(item))
        rows += got
        real = sum(r["truth"] != "none" for r in got)
        print(f"{item['id']:<28} {len(got)} candidates, {real} real plays", flush=True)
    (CAPTURES / "events.json").write_text(json.dumps(rows, indent=0))


if __name__ == "__main__":
    main()
