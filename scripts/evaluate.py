"""Score the helper's view of the opponent against hand labels.

    python scripts/evaluate.py                 # every labeled recording, deck known and learned
    python scripts/evaluate.py match1 match3   # some recordings
    python scripts/evaluate.py --live          # run the detector instead of the scan cache

For each recording it reports plays found, false plays, plays named right, and how far
the opponent elixir estimate is from the elixir the labeled plays imply, frame by frame.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

from clash_helper import kcr
from clash_helper.cards import card
from clash_helper.cli import video_frames
from clash_helper.game import Match
from clash_helper.tracker import UNKNOWN, Opponent

CAPTURES = Path(__file__).resolve().parents[1] / "captures"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
SAME_SECONDS = 1.5
SAME_PIXELS = 130
SPELL_EARLY = 3.5  # seconds a spell can be seen before its impact, a Goblin Barrel flies about 2.7
SPELL_LATE = 2.0  # labels mark a spell's launch, one known only by its impact is seen up to about 2 s later


class CachedDetector:
    """Plays back a scan, the newest analysed frame at or before the current time."""

    def __init__(self, rows: list) -> None:
        self.rows = rows
        self.at = 0
        self.busy = False

    def submit(self, frame, now: float) -> bool:
        while self.at + 1 < len(self.rows) and self.rows[self.at + 1][0] <= now + 1e-6:
            self.at += 1
        return True

    def result(self) -> tuple[float, list[kcr.Unit]]:
        t, units = self.rows[self.at]
        return t, [kcr.Unit(n, c, bool(e), conf, x, top, bottom, track) for n, c, e, conf, x, top, bottom, track in units]

    def close(self) -> None:
        pass


def truth_plays(labels: dict) -> list[dict]:
    if "plays" in labels:
        return labels["plays"]
    # the first labeling format: deploy events with verdicts plus the plays they missed
    plays = [{"time": e["time"], "x": e["x"], "y": e["y"], "card": e["label"]}
             for e in labels["events"] if e["label"] not in ("not_a_deploy", "unsure")]
    plays += [{"time": m["time"], "x": m.get("x", -1), "y": m.get("y", -1), "card": m["label"]} for m in labels.get("missed", [])]
    return sorted(plays, key=lambda p: p["time"])


def observations(item: dict, match: Match, live: bool):
    """Cached observations when research/observe.py made them, else the video frame by frame."""
    cached = CAPTURES / "obs" / f"{item['id']}.json.gz"
    if not live and cached.exists():
        from observe import load

        for obs in load(cached):
            yield obs
        return
    for frame, now in video_frames(Path(item["video"]), 6):
        yield match.observe(frame, now)


def pair(truth: list[dict], plays: list[tuple[float, int, int, bool]]) -> list[int | None]:
    """The labeled play each (time, x, y, is spell) play stands for, greedily in order. Spells are
    judged by time alone since they are seen far from where they land, troops by time and place."""
    used: set[int] = set()
    out: list[int | None] = []
    for t, x, y, spell in plays:
        hit = None
        for i, p in enumerate(truth):
            if i in used or card(p["card"]).spell != spell:
                continue
            early, late = (SPELL_EARLY, SPELL_LATE) if spell else (SAME_SECONDS, SAME_SECONDS)
            if not -early <= t - p["time"] <= late:
                continue
            if not spell and p.get("x", -1) >= 0 and abs(p["x"] - x) + abs(p["y"] - y) > SAME_PIXELS:
                continue
            hit = i
            break
        if hit is not None:
            used.add(hit)
        out.append(hit)
    return out


def run(item: dict, deck_known: bool, live: bool) -> tuple[list, list, float | None]:
    scan = CAPTURES / "scans" / f"{item['id']}.json.gz"
    if live or not scan.exists():
        detector = kcr.Detector()
    else:
        with gzip.open(scan, "rt") as f:
            detector = CachedDetector(json.load(f))
    match = Match(detector=detector, lookup=False, sync=True)
    plays: list = []
    elixir: list[tuple[float, float, float]] = []  # time, estimate, multiplier
    start = None
    shown: dict[int, float] = {}  # play -> when it first showed up
    for obs in observations(item, match, live):
        now = obs.now
        match.update(obs)
        if deck_known and match.started is not None and not match.deck_from_api:
            # stands in for a successful lookup of their deck
            match.opp.preload(item["their_deck"])
            match.deck_from_api = True
        if match.started is not None:
            start = match.started
            elixir.append((now, match.their_elixir, match.multiplier))
        # names keep improving as evidence arrives, keep the latest before the match resets
        if match.debug:
            for p, _, name in match.debug:
                if name != "none":
                    shown.setdefault(id(p), now)
            # candidates read as no play were dropped by the helper
            plays = [(p.time, p.x, p.y, name, p.source, shown[id(p)]) for p, _, name in match.debug if name != "none"]
    detector.close()
    return plays, elixir, start


def true_elixir(plays: list[dict], elixir: list, start: float | None) -> list[float]:
    """The elixir model fed with the labeled plays over the same clock."""
    if start is None:
        return []
    opp = Opponent(last_time=start)
    queue = sorted(plays, key=lambda p: p["time"])
    out, i = [], 0
    for now, _, multiplier in elixir:
        while i < len(queue) and queue[i]["time"] <= now:
            name = queue[i]["card"]
            opp.play(name, card(name).cost, max(start, queue[i]["time"]), multiplier)
            i += 1
        opp.tick(now, multiplier)
        out.append(opp.elixir)
    return out


def score(item: dict, labels: dict, deck_known: bool, live: bool) -> dict:
    truth = truth_plays(labels)
    plays, elixir, start = run(item, deck_known, live)
    hits = pair(truth, [(t, x, y, source == "spell") for t, x, y, _, source, _ in plays])
    used = {h for h in hits if h is not None}
    right = wrong = unknown = 0
    errors = []
    found = {"false": [], "wrong": [], "missed": []}
    delays = []
    for (t, x, y, name, source, shown), hit in zip(plays, hits):
        if hit is None:
            errors.append(f"false  {t:6.1f}s ({x},{y}) {name} from {source}")
            found["false"].append((name, source))
            continue
        delays.append(shown - truth[hit]["time"])
        label = truth[hit]["card"]
        if name == label:
            right += 1
        elif name == UNKNOWN:
            unknown += 1
            errors.append(f"?      {t:6.1f}s ({x},{y}) was {label}")
        else:
            wrong += 1
            errors.append(f"wrong  {t:6.1f}s ({x},{y}) {name}, was {label}")
            found["wrong"].append((name, label))
    for i, p in enumerate(truth):
        if i not in used:
            errors.append(f"missed {p['time']:6.1f}s ({p.get('x')},{p.get('y')}) {p['card']}")
            found["missed"].append(p["card"])
    estimate = [e for _, e, _ in elixir]
    actual = true_elixir(truth, elixir, start)
    gap = sum(abs(a - b) for a, b in zip(estimate, actual)) / len(actual) if actual else float("nan")
    return {"id": item["id"], "truth": len(truth), "found": len(used), "false": len(plays) - len(used),
            "right": right, "wrong": wrong, "unknown": unknown, "elixir_error": round(gap, 2), "errors": errors,
            "delay": round(sorted(delays)[len(delays) // 2], 2) if delays else float("nan"),
            "spells": sum(card(p["card"]).spell for p in truth),
            "spells_found": sum(card(truth[i]["card"]).spell for i in used), "breakdown": found}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--known-only", action="store_true")
    args = parser.parse_args()
    items = [i for i in json.loads((CAPTURES / "dataset.json").read_text()) if i["labels"]]
    items = [i for i in items if not args.ids or i["id"] in args.ids]
    for deck_known in (True,) if args.known_only else (True, False):
        print(f"\n== opponent deck {'known' if deck_known else 'learned from play'}")
        print(f"{'recording':<26}{'found':>9}{'false':>7}{'named':>9}{'spells':>9}{'elixir err':>12}")
        totals = dict.fromkeys(("truth", "found", "false", "right", "spells", "spells_found"), 0)
        gaps = []
        for item in items:
            r = score(item, json.loads(Path(item["labels"]).read_text()), deck_known, args.live)
            for k in totals:
                totals[k] += r[k]
            gaps.append(r["elixir_error"])
            print(f"{r['id']:<26}{r['found']:>4}/{r['truth']:<4}{r['false']:>7}{r['right']:>5}/{r['found']:<3}"
                  f"{r['spells_found']:>5}/{r['spells']:<3}{r['elixir_error']:>12}", flush=True)
            if args.verbose:
                print("   " + "\n   ".join(r["errors"]))
        t = totals
        print(f"{'all':<26}{t['found']:>4}/{t['truth']:<4}{t['false']:>7}{t['right']:>5}/{t['found']:<3}"
              f"{t['spells_found']:>5}/{t['spells']:<3}{sum(gaps) / len(gaps):>12.2f}")


if __name__ == "__main__":
    main()
