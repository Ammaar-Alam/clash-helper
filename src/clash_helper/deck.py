"""Pick a deck from real battles near the player's trophies that the player can build."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from clash_helper import api
from clash_helper.cards import card

MIN_PLAYERS = 3  # one hot streak should not crown a deck
PER_PLAYER = 5  # games one player can add to a deck's record
PLAYERS = 400  # battle logs to read, each holds about 30 matches
LEVEL_GAP = 1.5  # skip decks this far under the player's usual card level


@dataclass
class Pick:
    cards: tuple[str, ...]
    games: int
    wins: int
    level: float  # the player's average card level in this deck
    evolved: list[str]
    players: int = 0
    usual_level: float = 0  # the player's average level over their best cards

    @property
    def win_rate(self) -> float:
        # shrink small samples toward a coin flip, 10 imaginary games at 50 percent
        return (self.wins + 5) / (self.games + 10)

    @property
    def cost(self) -> float:
        return sum(card(c).cost for c in self.cards) / len(self.cards)

    @property
    def score(self) -> float:
        # each card level is roughly a tenth more stats, a tower attacker is what the timing calls send in
        attacker = any(card(c).has("win") for c in self.cards)
        return self.win_rate + 0.08 * (self.level - self.usual_level) + (0.02 if attacker else 0)


def _display_level(c: dict) -> int:
    return c["level"] + 16 - c["maxLevel"]


def _trophy_battles(tag: str) -> list[dict]:
    try:
        return [b for b in api.battlelog(tag) if b.get("type") == "PvP"]
    except api.ApiError:
        return []


def collect(tag: str) -> list[dict]:
    """Trophy road battles from the player's opponents and their opponents."""
    seen = {tag.lstrip("#").upper()}
    frontier = [tag]
    read = 0
    battles: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        while frontier and read < PLAYERS:
            frontier = frontier[: PLAYERS - read]
            read += len(frontier)
            logs = list(pool.map(_trophy_battles, frontier))
            frontier = []
            for log in logs:
                for battle in log:
                    battles.append(battle)
                    opponent = battle["opponent"][0]["tag"].lstrip("#")
                    if opponent not in seen:
                        seen.add(opponent)
                        frontier.append(opponent)
    # every battle appears in both players' logs once collected that deep
    unique = {(b["battleTime"], *sorted(p["tag"] for p in b["team"] + b["opponent"])): b for b in battles}
    return list(unique.values())


def rank(player: dict, battles: list[dict]) -> list[Pick]:
    me = player["tag"]
    owned = {c["name"]: c for c in player["cards"]}
    best = sorted((_display_level(c) for c in owned.values()), reverse=True)[:16]
    usual = sum(best) / len(best)
    # deck -> player -> [games, wins]
    tally: dict[tuple[str, ...], dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for battle in battles:
        for side, other in (("team", "opponent"), ("opponent", "team")):
            who = battle[side][0]
            deck = tuple(sorted(c["name"] for c in who["cards"]))
            if who["tag"] == me or len(deck) != 8 or not all(name in owned for name in deck):
                continue
            record = tally[deck][who["tag"]]
            if record[0] < PER_PLAYER:
                record[0] += 1
                record[1] += int(who["crowns"] > battle[other][0]["crowns"])
    picks = [
        Pick(
            deck,
            sum(g for g, _ in by.values()),
            sum(w for _, w in by.values()),
            sum(_display_level(owned[c]) for c in deck) / 8,
            [c for c in deck if owned[c].get("evolutionLevel")],
            len(by),
            usual,
        )
        for deck, by in tally.items()
        if len(by) >= MIN_PLAYERS
    ]
    picks = [p for p in picks if p.level >= usual - LEVEL_GAP]
    return sorted(picks, key=lambda p: p.score, reverse=True)


def suggest(tag: str, top: int = 3) -> None:
    player = api.player(tag)
    print(f"reading battles around {player['trophies']} trophies...", flush=True)
    battles = collect(tag)
    picks = rank(player, battles)
    print(f"{len(battles)} battles, {len(picks)} decks you can build that at least {MIN_PLAYERS} players used\n")
    for i, p in enumerate(picks[:top], 1):
        print(f"{i}. {', '.join(p.cards)}")
        print(f"   won {p.wins} of {p.games} across {p.players} players, your levels average {p.level:.1f}, {p.cost:.1f} elixir")
        if p.evolved:
            print(f"   put {', '.join(p.evolved)} in the first slots to use your evolutions")
        print()
