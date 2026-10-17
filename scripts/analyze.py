"""What the recorded games say about the advice.

    python scripts/analyze.py

A call counts as followed when that card is played within a few seconds of
the call appearing. Rules that win more when followed are the ones to trust.
"""

from __future__ import annotations

import json
from collections import defaultdict

from clash_helper.journal import GAMES

FOLLOW_SECONDS = 5.0


def load() -> list[tuple[dict, list[dict]]]:
    games = []
    for folder in sorted(GAMES.glob("*")):
        summary, events = folder / "summary.json", folder / "events.jsonl"
        if summary.exists() and events.exists():
            games.append((json.loads(summary.read_text()), [json.loads(line) for line in events.read_text().splitlines()]))
    return games


def main() -> None:
    games = [(s, e) for s, e in load() if s.get("result")]
    if not games:
        print(f"no finished games with results in {GAMES} yet")
        return
    wins = sum(s["result"]["won"] for s, _ in games)
    print(f"{len(games)} games, won {wins} ({wins / len(games):.0%})\n")

    # rule -> [shown, followed, followed in wins, followed in losses]
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    follow_rates = []
    for summary, events in games:
        won = summary["result"]["won"]
        plays = [e for e in events if e["type"] == "my_play"]
        calls = [e for e in events if e["type"] == "call" and e.get("card")]
        followed = 0
        for c in calls:
            row = stats[c["rule"]]
            row[0] += 1
            if any(p["card"] == c["card"] and 0 <= p["t"] - c["t"] <= FOLLOW_SECONDS for p in plays):
                followed += 1
                row[1] += 1
                row[2 if won else 3] += 1
        follow_rates.append((followed / len(calls) if calls else 0, won))

    print(f"{'rule':<10}{'shown':>7}{'followed':>10}{'win when followed':>20}")
    for rule, (shown, followed, in_wins, in_losses) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
        rate = f"{in_wins / followed:.0%}" if followed else "-"
        print(f"{rule:<10}{shown:>7}{followed:>10}{rate:>20}")

    follow_rates.sort()
    half = len(follow_rates) // 2
    if half:
        low = sum(w for _, w in follow_rates[:half]) / half
        high = sum(w for _, w in follow_rates[half:]) / (len(follow_rates) - half)
        print(f"\nwin rate when following the least: {low:.0%}, the most: {high:.0%}")


if __name__ == "__main__":
    main()
