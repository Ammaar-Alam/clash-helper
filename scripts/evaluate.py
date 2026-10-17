"""Score enemy deploy detection and naming against hand labels.

    python scripts/evaluate.py match1            # deck known, as after a successful lookup
    python scripts/evaluate.py match1 --no-deck  # deck learned from plays only

Reads captures/<match>.mp4, captures/<match>_labels.json and captures/matches.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clash_helper.cli import video_frames
from clash_helper.game import Match

CAPTURES = Path(__file__).resolve().parents[1] / "captures"
SAME_SECONDS = 1.0
SAME_PIXELS = 80


def predictions(match_name: str, use_deck: bool) -> list[tuple[float, int, int, str]]:
    deck = json.loads((CAPTURES / "matches.json").read_text())[match_name]["deck"]
    match = Match(lookup=False)
    final: list = []
    for frame, now in video_frames(CAPTURES / f"{match_name}.mp4", 5):
        match.step(frame, now)
        if use_deck and match.started is not None and not match.opp.deck_complete:
            match.opp.preload(deck)
        # names keep improving as answers arrive, keep the latest before the match resets
        if match.debug:
            final = list(match.debug)
    return [(p.time, p.x, p.y, name) for p, _, name in final]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("match")
    parser.add_argument("--no-deck", action="store_true")
    args = parser.parse_args()
    labels = json.loads((CAPTURES / f"{args.match}_labels.json").read_text())
    truth = [(e["time"], e["x"], e["y"], e["label"]) for e in labels["events"] if e["label"] != "not_a_deploy"]
    truth += [(m["time"], m.get("x", -999), m.get("y", -999), m["label"]) for m in labels.get("missed", [])]
    preds = predictions(args.match, not args.no_deck)

    used: set[int] = set()
    named = unknown = wrong = 0
    for t, x, y, name in preds:
        hit = next(
            (
                i
                for i, (lt, lx, ly, _) in enumerate(truth)
                if i not in used and abs(lt - t) <= SAME_SECONDS and (lx < 0 or abs(lx - x) + abs(ly - y) <= SAME_PIXELS)
            ),
            None,
        )
        if hit is None:
            print(f"  false  {t:6.1f}s ({x},{y}) -> {name}")
            continue
        used.add(hit)
        label = truth[hit][3]
        if name == label:
            named += 1
        elif name == "?":
            unknown += 1
            print(f"  ?      {t:6.1f}s ({x},{y}) was {label}")
        else:
            wrong += 1
            print(f"  wrong  {t:6.1f}s ({x},{y}) -> {name}, was {label}")
    for i, (t, x, y, label) in enumerate(truth):
        if i not in used:
            print(f"  missed {t:6.1f}s ({x},{y}) {label}")

    matched = len(used)
    print(f"\n{args.match}, deck {'learned' if args.no_deck else 'known'}")
    print(f"deploys found {matched}/{len(truth)}, false alarms {len(preds) - matched}")
    print(f"named right {named}/{matched}, wrong {wrong}, unknown {unknown}")


if __name__ == "__main__":
    main()
