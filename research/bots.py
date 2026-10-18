"""Decks the ladder bots play, from battle logs around your trophies.

    python research/bots.py     # a few thousand API calls, several minutes

Most opponents near 4300 trophies are bots: top level, no experience or star points, about 30%
wins over tens of thousands of battles, and a clan name on screen the API does not know. They
play a fixed set of decks, so a few of their cards give away the rest. Walks from your battle
log through the real players you met to the bots they met. Writes src/clash_helper/data/bot_decks.json,
each deck as a sorted list of 8 card names with how many bot games showed it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clash_helper import api

OUT = Path(__file__).resolve().parents[1] / "src/clash_helper/data/bot_decks.json"
TROPHIES = 600
PLAYER_LOGS = 150


def is_bot(tag: str) -> bool:
    p = api.player(tag)
    return p.get("expPoints") == 0 and p.get("starPoints") is None and p.get("clanCardsCollected") == 0


def main() -> None:
    me = api.setting("CR_PLAYER_TAG").lstrip("#")
    mine = api.player(me)["trophies"]
    decks: Counter = Counter()
    checked: dict[str, bool] = {}
    queue, read = deque([me]), set()
    pool = ThreadPoolExecutor(6)
    while queue and len(read) < PLAYER_LOGS:
        tag = queue.popleft()
        if tag in read:
            continue
        read.add(tag)
        try:
            log = api.battlelog(tag)
        except api.ApiError as error:
            print(f"skipped {tag}: {error}", file=sys.stderr)
            continue
        games = [(o["tag"].lstrip("#"), tuple(sorted(c["name"] for c in o["cards"])))
                 for b in log if b.get("gameMode", {}).get("name") == "Ladder"
                 for o in b["opponent"] if abs(o.get("startingTrophies", 0) - mine) <= TROPHIES and len(o["cards"]) == 8]
        fresh = [t for t in {t for t, _ in games} if t not in checked]
        for t, bot in zip(fresh, pool.map(lambda t: _safe(is_bot, t), fresh)):
            checked[t] = bool(bot)
        for t, deck in games:
            if checked.get(t):
                decks[deck] += 1
            elif t not in read:
                queue.append(t)
        if len(read) % 10 == 0:
            print(f"{len(read)} logs, {len(checked)} players checked, {sum(checked.values())} bots, "
                  f"{len(decks)} bot decks from {sum(decks.values())} games", flush=True)
    OUT.write_text(json.dumps([[list(d), n] for d, n in decks.most_common()], indent=0))
    print(f"wrote {len(decks)} bot decks from {sum(decks.values())} bot games to {OUT.name}")


def _safe(f, *args):
    try:
        return f(*args)
    except api.ApiError:
        return None


if __name__ == "__main__":
    main()
