"""Where the remaining errors are, across every labeled recording.

    python research/report.py            # deck known
    python research/report.py --learned  # deck learned from play
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import CAPTURES, score  # noqa: E402

from clash_helper.cards import card  # noqa: E402


def main() -> None:
    known = "--learned" not in sys.argv
    items = [i for i in json.loads((CAPTURES / "dataset.json").read_text()) if i["labels"]]
    missed, false, wrong = Counter(), Counter(), Counter()
    totals = Counter()
    for item in items:
        r = score(item, json.loads(Path(item["labels"]).read_text()), known, False)
        for k in ("truth", "found", "false", "right", "spells", "spells_found"):
            totals[k] += r[k]
        totals["elixir"] += r["elixir_error"]
        missed.update(r["breakdown"]["missed"])
        false.update(f"{name} ({source})" for name, source in r["breakdown"]["false"])
        wrong.update(f"{a} for {b}" for a, b in r["breakdown"]["wrong"])
        print(f"{r['id']:<28} found {r['found']}/{r['truth']} false {r['false']} named {r['right']}/{r['found']} "
              f"spells {r['spells_found']}/{r['spells']} elixir {r['elixir_error']}", flush=True)
    t = totals
    print(f"\nall: found {t['found']}/{t['truth']} ({t['found'] / t['truth']:.0%}), false {t['false']}, "
          f"named {t['right']}/{t['found']} ({t['right'] / max(1, t['found']):.0%}), spells {t['spells_found']}/{t['spells']}, "
          f"mean elixir error {t['elixir'] / len(items):.2f}")
    spells = Counter({k: v for k, v in missed.items() if card(k).spell})
    print("\nmissed spells:", dict(spells.most_common()))
    print("missed troops:", dict(Counter({k: v for k, v in missed.items() if not card(k).spell}).most_common()))
    print("false plays:", dict(false.most_common(15)))
    print("wrong names:", dict(wrong.most_common(15)))


if __name__ == "__main__":
    main()
