"""Where the remaining errors are, across every labeled recording.

    python research/report.py                  # deck known
    python research/report.py --learned        # deck learned from play
    python research/report.py --detector-only  # without learned class meanings

Each recording is scored with class meanings learned from the other recordings only
(captures/events.json, from research/events.py), so the numbers hold for unseen games.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import CAPTURES, score  # noqa: E402
from train import EVENTS, fit  # noqa: E402

from clash_helper import bots, confusion  # noqa: E402
from clash_helper.cards import card  # noqa: E402

SHARP = 600  # px wide, the first helper recordings were saved at half resolution


def held_out(recording: str, rows: list[dict], folder: Path) -> None:
    """Point the helper at class meanings learned without this recording, and at bot decks without
    this game's own."""
    path = folder / f"{recording}.json"
    path.write_text(json.dumps(fit([r for r in rows if r["recording"] != recording])))
    os.environ["CLASH_CONFUSION"] = str(path)
    confusion.table.cache_clear()
    # the bot decks were gathered after this game, so it is in there once
    theirs = next(sorted(i["their_deck"]) for i in json.loads((CAPTURES / "dataset.json").read_text())
                  if i["id"] == recording)
    pool = [[d, n - (d == theirs)] for d, n in json.loads(bots.PATH.read_text())]
    decks = folder / f"{recording}_bots.json"
    decks.write_text(json.dumps([[d, n] for d, n in pool if n > 0]))
    os.environ["CLASH_BOT_DECKS"] = str(decks)
    bots.decks.cache_clear()


def summary(name: str, results: list[dict]) -> str:
    t = Counter()
    for r in results:
        for k in ("truth", "found", "false", "right", "spells", "spells_found"):
            t[k] += r[k]
    return (f"{name}: found {t['found']}/{t['truth']} ({t['found'] / t['truth']:.0%}), false {t['false']}, "
            f"named {t['right']}/{t['found']} ({t['right'] / max(1, t['found']):.0%}), "
            f"spells {t['spells_found']}/{t['spells']}, "
            f"mean elixir error {sum(r['elixir_error'] for r in results) / len(results):.2f}, "
            f"median delay {sorted(r['delay'] for r in results)[len(results) // 2]:.1f} s")


def main() -> None:
    known = "--learned" not in sys.argv
    rows = [] if "--detector-only" in sys.argv else json.loads(EVENTS.read_text())
    items = [i for i in json.loads((CAPTURES / "dataset.json").read_text()) if i["labels"]]
    missed, false, wrong = Counter(), Counter(), Counter()
    results, sharp = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for item in items:
            held_out(item["id"], rows, Path(tmp))
            r = score(item, json.loads(Path(item["labels"]).read_text()), known, False)
            results.append(r)
            if cv2.VideoCapture(item["video"]).get(cv2.CAP_PROP_FRAME_WIDTH) >= SHARP:
                sharp.append(r)
            missed.update(r["breakdown"]["missed"])
            false.update(f"{name} ({source})" for name, source in r["breakdown"]["false"])
            wrong.update(f"{a} for {b}" for a, b in r["breakdown"]["wrong"])
            print(f"{r['id']:<28} found {r['found']}/{r['truth']} false {r['false']} named {r['right']}/{r['found']} "
                  f"spells {r['spells_found']}/{r['spells']} elixir {r['elixir_error']} delay {r['delay']}", flush=True)
    print()
    print(summary("all", results))
    print(summary("full resolution", sharp))
    spells = Counter({k: v for k, v in missed.items() if card(k).spell})
    print("\nmissed spells:", dict(spells.most_common()))
    print("missed troops:", dict(Counter({k: v for k, v in missed.items() if not card(k).spell}).most_common()))
    print("false plays:", dict(false.most_common(15)))
    print("wrong names:", dict(wrong.most_common(15)))


if __name__ == "__main__":
    main()
