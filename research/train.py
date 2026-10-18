"""Learn what each detector class really means and measure it on held out recordings.

    python research/events.py     # refresh the labeled candidates first
    python research/train.py      # leave one recording out scores, then the table for the helper

Each held out recording is scored with a table learned from all the others, so the numbers
show how the helper does on a game it has never seen. The final table, learned from every
recording, is written to src/clash_helper/data/confusion.json for live use.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "captures/events.json"
OUT = ROOT / "src/clash_helper/data/confusion.json"


def fit(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Counts of the true card for each detector class, weighted by its share of the votes."""
    table: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        total = sum(r["votes"].values())
        for detected, weight in r["votes"].items():
            table[detected][r["truth"]] += weight / total if total else 0
    return {d: {t: round(v, 3) for t, v in c.items()} for d, c in table.items()}


def evaluate(recording: str, table_path: str | None) -> str:
    env = {**os.environ}
    if table_path:
        env["CLASH_CONFUSION"] = table_path
    out = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), recording], env=env,
                         capture_output=True, text=True, cwd=ROOT).stdout
    return out


def main() -> None:
    rows = json.loads(EVENTS.read_text())
    recordings = sorted({r["recording"] for r in rows})
    lines: dict[str, list[str]] = defaultdict(list)
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty.json"
        empty.write_text("{}")
        for rec in recordings:
            held_out = Path(tmp) / f"{rec}.json"
            held_out.write_text(json.dumps(fit([r for r in rows if r["recording"] != rec])))
            for name, path in (("detector only", str(empty)), ("learned", str(held_out))):
                for line in evaluate(rec, path).splitlines():
                    if line.startswith(rec) or line.startswith("=="):
                        lines[name].append(line)
    for name, got in lines.items():
        print(f"\n### {name}")
        print("\n".join(got))
    OUT.write_text(json.dumps(fit(rows), indent=1, sort_keys=True))
    print(f"\nwrote {OUT.relative_to(ROOT)} from {len(rows)} candidates in {len(recordings)} recordings")


if __name__ == "__main__":
    main()
