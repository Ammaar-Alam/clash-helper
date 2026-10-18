"""Learn what each detector class really means from the labeled recordings.

    python research/events.py     # refresh the labeled candidates first
    python research/train.py      # write the table the helper uses
    python research/report.py     # held out scores, each recording judged by a table learned without it

Writes src/clash_helper/data/confusion.json, learned from every recording, for live use.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import cache  # noqa: E402

EVENTS = cache("events.json")
OUT = ROOT / "src/clash_helper/data/confusion.json"


def fit(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Counts of the true card for each detector class, weighted by its share of the votes."""
    table: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        total = sum(r["votes"].values())
        for detected, weight in r["votes"].items():
            table[detected][r["truth"]] += weight / total if total else 0
    return {d: {t: round(v, 3) for t, v in c.items()} for d, c in table.items()}


def main() -> None:
    rows = json.loads(EVENTS.read_text())
    OUT.write_text(json.dumps(fit(rows), indent=1, sort_keys=True))
    recordings = {r["recording"] for r in rows}
    print(f"wrote {OUT.relative_to(ROOT)} from {len(rows)} candidates in {len(recordings)} recordings")


if __name__ == "__main__":
    main()
