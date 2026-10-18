"""How well the Zap bolt and Arrows ring signals find labeled spells, across thresholds.

    python research/spell_signals.py

A peak counts as a hit when it falls within a second or so of a labeled spell of that kind.
Reports, per threshold, the share of labeled spells found and false peaks per minute of play.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import CAPTURES, truth_plays  # noqa: E402
from observe import load  # noqa: E402

SIGNALS = {"Zap": ("bolt", 0), "Arrows": ("ring", 0)}
WINDOW = (-0.6, 1.6)  # seconds from the labeled first frame the signal may peak


def peaks(series: list[tuple[float, float]], threshold: float, gap: float = 1.5) -> list[float]:
    """Times where the signal crosses the threshold, one per burst."""
    out, last = [], -9.0
    for t, v in series:
        if v >= threshold and t - last > gap:
            out.append(t)
            last = t
        elif v >= threshold:
            last = t
    return out


def main() -> None:
    items = [i for i in json.loads((CAPTURES / "dataset.json").read_text())
             if i["labels"] and (CAPTURES / "obs" / f"{i['id']}.json.gz").exists()]
    data = []
    for item in items:
        obs = [o for o in load(CAPTURES / "obs" / f"{item['id']}.json.gz") if o.in_match]
        truth = truth_plays(json.loads(Path(item["labels"]).read_text()))
        minutes = (obs[-1].now - obs[0].now) / 60 if obs else 0
        data.append((item["id"], obs, truth, minutes))
    for spell, (field, index) in SIGNALS.items():
        labeled = sum(1 for _, _, truth, _ in data for p in truth if p["card"] == spell)
        print(f"\n== {spell}: {labeled} labeled across {len(data)} recordings")
        for threshold in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6):
            hit = false = 0
            minutes = 0.0
            for _, obs, truth, mins in data:
                minutes += mins
                found = peaks([(o.now, getattr(o, field)[index]) for o in obs], threshold)
                times = [p["time"] for p in truth if p["card"] == spell]
                used = set()
                for t in found:
                    match = next((i for i, s in enumerate(times) if i not in used and WINDOW[0] <= t - s <= WINDOW[1]), None)
                    if match is None:
                        false += 1
                    else:
                        used.add(match)
                hit += len(used)
            print(f"   threshold {threshold:<5} found {hit}/{labeled}   false {false} ({false / max(minutes, 1e-9):.2f} per minute)")


if __name__ == "__main__":
    main()
