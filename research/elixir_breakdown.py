"""Which kind of error the opponent elixir error comes from.

    python research/elixir_breakdown.py

Starts from the live estimate, as the helper knew it frame by frame, then feeds the elixir model the
final plays with one kind of error repaired at a time: every play known the moment it happened, wrong
names fixed, missed plays added, false plays removed. Tables are held out as in research/report.py.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import CAPTURES, pair, run, true_elixir, truth_plays  # noqa: E402
from report import held_out  # noqa: E402
from train import EVENTS  # noqa: E402

from clash_helper.tracker import UNKNOWN  # noqa: E402

STEPS = ("live", "known at once", "names fixed", "+ missed", "+ false removed")


def gap(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / max(1, len(a))


def main() -> None:
    rows = json.loads(EVENTS.read_text())
    items = [i for i in json.loads((CAPTURES / "dataset.json").read_text()) if i["labels"]]
    sums: Counter = Counter()
    with tempfile.TemporaryDirectory() as tmp:
        for item in items:
            held_out(item["id"], rows, Path(tmp))
            truth = truth_plays(json.loads(Path(item["labels"]).read_text()))
            plays, elixir, start = run(item, True, False)
            hits = pair(truth, [(t, x, y, source == "spell") for t, x, y, _, source, _ in plays])
            actual = true_elixir(truth, elixir, start)
            # unnamed plays are left out of the replays, the live model counts them as 3 elixir
            at_once = [{"time": p[0], "card": p[3]} for p in plays if p[3] != UNKNOWN]
            fixed = [{"time": p[0], "card": truth[h]["card"]} if h is not None else {"time": p[0], "card": p[3]}
                     for p, h in zip(plays, hits) if h is not None or p[3] != UNKNOWN]
            missed = [truth[i] for i in range(len(truth)) if i not in set(hits)]
            real = [{"time": p[0], "card": truth[h]["card"]} for p, h in zip(plays, hits) if h is not None]
            curves = [[e for _, e, _ in elixir]] + [true_elixir(c, elixir, start) for c in
                                                    (at_once, fixed, fixed + missed, real + missed)]
            row = [gap(c, actual) for c in curves]
            for name, v in zip(STEPS, row):
                sums[name] += v
            print(f"{item['id']:<28} " + "  ".join(f"{n} {v:.2f}" for n, v in zip(STEPS, row)), flush=True)
    print("\nmean: " + ", ".join(f"{n} {sums[n] / len(items):.2f}" for n in STEPS))


if __name__ == "__main__":
    main()
