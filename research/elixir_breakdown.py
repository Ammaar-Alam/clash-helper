"""Which kind of error the opponent elixir error comes from.

Feeds the elixir model the helper's plays with one kind of error repaired at a time:
wrong names fixed, then missed plays added, then false plays removed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import CAPTURES, SAME_PIXELS, SAME_SECONDS, SPELL_EARLY, run, true_elixir, truth_plays  # noqa: E402

from clash_helper.cards import card  # noqa: E402


def curve(plays: list[dict], elixir: list, start: float) -> list[float]:
    return true_elixir(plays, elixir, start)


def gap(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / max(1, len(a))


def main() -> None:
    items = [i for i in json.loads((CAPTURES / "dataset.json").read_text()) if i["labels"]]
    sums = [0.0] * 5
    for item in items:
        truth = truth_plays(json.loads(Path(item["labels"]).read_text()))
        plays, elixir, start = run(item, True, False)
        if start is None:
            continue
        pairs, used = [], set()
        for t, x, y, name, _ in plays:
            hit = None
            for i, p in enumerate(truth):
                if i in used:
                    continue
                spell = card(p["card"]).spell
                early = SPELL_EARLY if spell else SAME_SECONDS
                if not (-early <= t - p["time"] <= SAME_SECONDS):
                    continue
                if not spell and p.get("x", -1) >= 0 and abs(p["x"] - x) + abs(p["y"] - y) > SAME_PIXELS:
                    continue
                hit = i
                break
            if hit is not None:
                used.add(hit)
            pairs.append((t, name, hit))
        actual = curve(truth, elixir, start)
        known = lambda n: n not in ("?", "none")  # noqa: E731
        as_seen = [{"time": t, "card": n} for t, n, _ in pairs if known(n)]
        names_fixed = [{"time": t, "card": truth[h]["card"] if h is not None else n} for t, n, h in pairs
                       if h is not None or known(n)]
        with_missed = names_fixed + [truth[i] for i in range(len(truth)) if i not in used]
        no_false = [p for p in with_missed if not (p.get("card") and p not in truth and
                    not any(abs(p["time"] - q["time"]) < 1.6 for q in truth))]
        timing_only = [{"time": t, "card": truth[h]["card"]} for t, n, h in pairs if h is not None] + \
                      [truth[i] for i in range(len(truth)) if i not in used]
        row = [gap(curve(as_seen, elixir, start), actual), gap(curve(names_fixed, elixir, start), actual),
               gap(curve(with_missed, elixir, start), actual), gap(curve(no_false, elixir, start), actual),
               gap(curve(timing_only, elixir, start), actual)]
        sums = [a + b for a, b in zip(sums, row)]
        print(f"{item['id']:<28} as seen {row[0]:.2f}  names fixed {row[1]:.2f}  + missed added {row[2]:.2f}  "
              f"+ false removed {row[3]:.2f}  (only timing left {row[4]:.2f})", flush=True)
    n = len(items)
    print(f"\nmean: as seen {sums[0] / n:.2f}, names fixed {sums[1] / n:.2f}, + missed {sums[2] / n:.2f}, "
          f"+ false removed {sums[3] / n:.2f}, timing only {sums[4] / n:.2f}")


if __name__ == "__main__":
    main()
