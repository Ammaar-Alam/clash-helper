"""Card facts the advice rules reason with.

Costs and types come from data/cards.json (scripts/build_cards.py). Roles are
hand-written because no data source states what a card is for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

DATA = Path(__file__).parent / "data" / "cards.json"

# win: goes for towers, air: flies, aa: hits air, splash: area damage,
# swarm: several small units, tank: soaks damage, dps: shreds big units,
# small/big: spell size, champion: leaves the cycle while alive
ROLES: dict[str, str] = {
    "Archer Queen": "aa champion",
    "Archers": "aa",
    "Arrows": "small",
    "Baby Dragon": "air aa splash",
    "Balloon": "air win",
    "Bandit": "",
    "Barbarian Barrel": "small",
    "Barbarian Hut": "",
    "Barbarians": "swarm",
    "Bats": "air aa swarm",
    "Battle Healer": "",
    "Battle Ram": "win",
    "Berserker": "",
    "Bomb Tower": "splash",
    "Bomber": "splash",
    "Boss Bandit": "champion",
    "Bowler": "splash",
    "Cannon": "",
    "Cannon Cart": "",
    "Dark Prince": "splash",
    "Dart Goblin": "aa",
    "Earthquake": "small",
    "Electro Dragon": "air aa splash",
    "Electro Giant": "tank win",
    "Electro Spirit": "aa",
    "Electro Wizard": "aa",
    "Elite Barbarians": "dps",
    "Elixir Collector": "",
    "Elixir Golem": "tank win",
    "Executioner": "aa splash",
    "Fire Spirit": "aa splash",
    "Fireball": "big",
    "Firecracker": "aa splash",
    "Fisherman": "",
    "Flying Machine": "air aa",
    "Freeze": "small",
    "Furnace": "aa",
    "Giant": "tank win",
    "Giant Skeleton": "tank splash",
    "Giant Snowball": "small",
    "Goblin Barrel": "win swarm",
    "Goblin Cage": "",
    "Goblin Demolisher": "splash",
    "Goblin Drill": "win",
    "Goblin Gang": "aa swarm",
    "Goblin Giant": "tank win aa",
    "Goblin Hut": "",
    "Goblin Machine": "",
    "Goblins": "swarm",
    "Goblinstein": "tank champion",
    "Golden Knight": "champion",
    "Golem": "tank win",
    "Graveyard": "win swarm",
    "Guards": "swarm",
    "Heal Spirit": "aa",
    "Hog Rider": "win",
    "Hunter": "aa dps",
    "Ice Golem": "tank",
    "Ice Spirit": "aa",
    "Ice Wizard": "aa splash",
    "Inferno Dragon": "air aa dps",
    "Inferno Tower": "aa dps",
    "Knight": "",
    "Lava Hound": "air tank win",
    "Lightning": "big",
    "Little Prince": "aa champion",
    "Lumberjack": "dps",
    "Magic Archer": "aa",
    "Mega Knight": "tank splash",
    "Mega Minion": "air aa",
    "Mighty Miner": "dps champion",
    "Miner": "win",
    "Mini P.E.K.K.A": "dps",
    "Minion Horde": "air aa swarm",
    "Minions": "air aa swarm",
    "Mirror": "",
    "Monk": "champion",
    "Mortar": "win splash",
    "Mother Witch": "aa",
    "Musketeer": "aa",
    "Night Witch": "",
    "P.E.K.K.A": "tank dps",
    "Phoenix": "air aa",
    "Poison": "big",
    "Prince": "dps",
    "Princess": "aa splash",
    "Rage": "",
    "Ram Rider": "win",
    "Rascals": "aa",
    "Rocket": "big",
    "Royal Delivery": "small",
    "Royal Ghost": "splash",
    "Royal Giant": "tank win",
    "Royal Hogs": "win swarm",
    "Royal Recruits": "swarm",
    "Rune Giant": "tank",
    "Skeleton Army": "swarm",
    "Skeleton Barrel": "win swarm",
    "Skeleton Dragons": "air aa splash",
    "Skeleton King": "splash champion",
    "Skeletons": "swarm",
    "Sparky": "splash",
    "Spear Goblins": "aa swarm",
    "Spirit Empress": "aa",
    "Suspicious Bush": "win",
    "Tesla": "aa",
    "The Log": "small",
    "Three Musketeers": "aa",
    "Tombstone": "",
    "Tornado": "small",
    "Valkyrie": "splash",
    "Void": "big",
    "Wall Breakers": "win",
    "Witch": "aa splash",
    "Wizard": "aa splash",
    "X-Bow": "win aa",
    "Zap": "small",
    "Zappies": "aa",
}

# detector unit names that differ from the card name
UNITS: dict[str, list[str]] = {
    "Archers": ["archer"],
    "Baby Dragon": ["hungry_dragon"],
    "Barbarian Barrel": ["barbarian"],
    "Barbarian Hut": ["barbarian_hut", "barbarian"],
    "Barbarians": ["barbarian"],
    "Bats": ["bat"],
    "Elite Barbarians": ["elite_barbarian"],
    "Elixir Golem": ["elixir_golem_large", "elixir_golem_medium", "elixir_golem_small"],
    "Goblin Barrel": ["goblin"],
    "Goblin Gang": ["goblin", "spear_goblin"],
    "Goblin Hut": ["goblin_hut", "spear_goblin"],
    "Goblins": ["goblin"],
    "Golem": ["golem", "golemite"],
    "Graveyard": ["skeleton"],
    "Guards": ["guard"],
    "Hog Rider": ["hog_rider"],
    "Lava Hound": ["lava_hound", "lava_pup"],
    "Little Prince": ["little_prince", "royal_guardian"],
    "Mini P.E.K.K.A": ["minipekka"],
    "Minion Horde": ["minion"],
    "Minions": ["minion"],
    "Mother Witch": ["mother_witch"],
    "P.E.K.K.A": ["pekka"],
    "Phoenix": ["phoenix_large", "phoenix_small", "phoenix_egg"],
    "Rascals": ["rascal_boy", "rascal_girl"],
    "Royal Hogs": ["royal_hog"],
    "Royal Recruits": ["royal_recruit"],
    "Skeleton Army": ["skeleton"],
    "Skeleton Barrel": ["skeleton"],
    "Skeleton Dragons": ["skeleton_dragon"],
    "Skeletons": ["skeleton"],
    "Spear Goblins": ["spear_goblin"],
    "Three Musketeers": ["musketeer"],
    "Tombstone": ["tombstone", "skeleton"],
    "Wall Breakers": ["wall_breaker"],
    "Witch": ["witch", "skeleton"],
    "Zappies": ["zappy"],
}


@dataclass(frozen=True)
class Card:
    name: str
    cost: int
    type: str
    roles: frozenset[str]

    def has(self, role: str) -> bool:
        return role in self.roles

    @property
    def spell(self) -> bool:
        return self.type == "spell"

    @property
    def building(self) -> bool:
        return self.type == "building"

    @property
    def units(self) -> list[str]:
        """Detector class names this card puts on the arena."""
        if self.spell and self.name not in UNITS:
            return []
        snake = self.name.lower().replace(".", "").replace("-", "_").replace(" ", "_")
        return UNITS.get(self.name, [snake])


@cache
def all_cards() -> dict[str, Card]:
    rows = json.loads(DATA.read_text())
    return {
        r["name"]: Card(r["name"], int(r["cost"] or 0), r["type"], frozenset(ROLES.get(r["name"], "").split()))
        for r in rows
    }


def card(name: str) -> Card:
    known = all_cards().get(name)
    if known:
        return known
    # a card newer than the data file still gets a name, its cost is unknown
    return Card(name, 0, "troop", frozenset(ROLES.get(name, "").split()))


def answers(threat: Card, answer: Card) -> bool:
    """Whether `answer` is a sensible response to `threat`."""
    if answer.spell:
        if threat.has("swarm"):
            return True
        return answer.has("big") and threat.cost <= 4 and not threat.has("tank")
    if threat.has("air") and not answer.has("aa"):
        return False
    if threat.has("swarm"):
        return answer.has("splash") or answer.has("swarm")
    if threat.has("tank"):
        return answer.has("dps") or answer.has("swarm") or answer.building
    if threat.has("win"):
        return answer.building or answer.has("swarm") or answer.has("dps") or answer.cost >= 3
    return answer.cost >= 3 or answer.has("swarm")
