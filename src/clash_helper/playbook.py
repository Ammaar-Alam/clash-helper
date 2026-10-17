"""Specific answers and placements, the knowledge a good player uses without thinking.

Each threat lists answers best first with where to put them. The advice rules
try these before falling back to the rough role matching in cards.answers.
"""

from __future__ import annotations

from clash_helper.board import LANE_COL, row_of, tile

# where an answer goes, filled with the lane and the threat at call time
SPOTS = {
    "pull": "center, 3 tiles from the river",
    "front": "in front of your {lane} tower",
    "on": "on the {threat}",
    "path": "{lane} lane, in its path",
    "behind": "behind your {lane} tower",
    "bridge": "{lane} bridge as it crosses",
    "back": "behind your king tower, {lane} side",
}

PULL_BUILDINGS = [("Cannon", "pull"), ("Tesla", "pull"), ("Inferno Tower", "pull"), ("Bomb Tower", "pull"), ("Tombstone", "pull"), ("Goblin Cage", "pull")]
SWARM_KILLERS = [("The Log", "on"), ("Arrows", "on"), ("Zap", "on"), ("Giant Snowball", "on"), ("Barbarian Barrel", "on"), ("Valkyrie", "on"), ("Baby Dragon", "path"), ("Bomber", "behind"), ("Wizard", "behind"), ("Witch", "behind"), ("Executioner", "behind"), ("Dark Prince", "path"), ("Fire Spirit", "path")]
AIR_SWARM_KILLERS = [("Arrows", "on"), ("Zap", "on"), ("Baby Dragon", "path"), ("Wizard", "behind"), ("Witch", "behind"), ("Executioner", "behind"), ("Fire Spirit", "path"), ("Electro Spirit", "path"), ("Archers", "behind"), ("Musketeer", "behind"), ("Fireball", "on")]
TANK_KILLERS = [("Inferno Tower", "pull"), ("Inferno Dragon", "path"), ("Mini P.E.K.K.A", "front"), ("P.E.K.K.A", "front"), ("Hunter", "front"), ("Prince", "front"), ("Lumberjack", "front"), ("Skeleton Army", "front"), ("Guards", "front"), ("Barbarians", "front"), ("Cannon", "pull"), ("Tesla", "pull")]
AIR_TANK_KILLERS = [("Inferno Tower", "pull"), ("Inferno Dragon", "path"), ("Musketeer", "behind"), ("Hunter", "front"), ("Mega Minion", "path"), ("Minions", "path"), ("Minion Horde", "path"), ("Archers", "behind"), ("Electro Wizard", "behind"), ("Tesla", "pull"), ("Witch", "behind"), ("Baby Dragon", "path")]
MINI_TANK_KILLERS = [("Mini P.E.K.K.A", "front"), ("Knight", "front"), ("Valkyrie", "front"), ("Skeleton Army", "front"), ("Guards", "front"), ("Cannon", "pull"), ("Tesla", "pull"), ("Prince", "front"), ("Lumberjack", "front"), ("P.E.K.K.A", "front"), ("Barbarians", "front")]
RANGED_KILLERS = [("Fireball", "on"), ("Mini P.E.K.K.A", "path"), ("Knight", "path"), ("Valkyrie", "path"), ("Bandit", "path"), ("Royal Ghost", "path"), ("Lumberjack", "path"), ("Miner", "on"), ("Baby Dragon", "path"), ("Mega Minion", "path")]
BARREL_KILLERS = [("The Log", "on"), ("Barbarian Barrel", "on"), ("Arrows", "on"), ("Zap", "on"), ("Giant Snowball", "on"), ("Valkyrie", "front"), ("Baby Dragon", "front"), ("Witch", "front"), ("Fire Spirit", "front")]

ANSWERS: dict[str, list[tuple[str, str]]] = {
    # buildings pull these off the towers, a tank killer finishes them
    "Hog Rider": PULL_BUILDINGS + [("Mini P.E.K.K.A", "bridge"), ("Knight", "bridge"), ("Valkyrie", "bridge"), ("Skeleton Army", "front"), ("Guards", "front"), ("Skeletons", "front"), ("Ice Spirit", "bridge")],
    "Royal Hogs": [("Valkyrie", "path"), ("Baby Dragon", "path"), ("Arrows", "on"), ("Fireball", "on"), ("Bomb Tower", "pull"), ("Cannon", "pull"), ("Witch", "path")],
    "Battle Ram": PULL_BUILDINGS + [("Mini P.E.K.K.A", "bridge"), ("Knight", "bridge"), ("Skeleton Army", "bridge")],
    "Ram Rider": PULL_BUILDINGS + [("Mini P.E.K.K.A", "bridge"), ("Knight", "bridge"), ("Skeleton Army", "front")],
    "Wall Breakers": [("The Log", "on"), ("Arrows", "on"), ("Zap", "on"), ("Cannon", "pull"), ("Skeletons", "front"), ("Archers", "behind"), ("Valkyrie", "front")],
    "Giant": [("Mini P.E.K.K.A", "front"), ("Cannon", "pull"), ("Inferno Tower", "pull"), ("Skeleton Army", "front"), ("P.E.K.K.A", "front"), ("Guards", "front"), ("Barbarians", "front")],
    "Royal Giant": [("Inferno Tower", "pull"), ("Cannon", "pull"), ("Mini P.E.K.K.A", "front"), ("Skeleton Army", "front"), ("P.E.K.K.A", "front"), ("Barbarians", "front"), ("Hunter", "front")],
    "Golem": TANK_KILLERS,
    "Electro Giant": [("Inferno Tower", "pull"), ("P.E.K.K.A", "front"), ("Mini P.E.K.K.A", "front"), ("Skeleton Army", "front"), ("Barbarians", "front"), ("Cannon", "pull")],
    "Goblin Giant": TANK_KILLERS,
    "Elixir Golem": TANK_KILLERS,
    "Giant Skeleton": [("Skeleton Army", "front"), ("Mini P.E.K.K.A", "front"), ("P.E.K.K.A", "front"), ("Cannon", "pull"), ("Inferno Tower", "pull")],
    "P.E.K.K.A": [("Inferno Tower", "pull"), ("Skeleton Army", "front"), ("Guards", "front"), ("Barbarians", "front"), ("Minions", "path"), ("Minion Horde", "path"), ("Cannon", "pull")],
    "Mega Knight": [("Inferno Tower", "pull"), ("P.E.K.K.A", "front"), ("Mini P.E.K.K.A", "front"), ("Minions", "path"), ("Minion Horde", "path"), ("Baby Dragon", "path"), ("Cannon", "pull"), ("Knight", "front")],
    "Balloon": [("Musketeer", "behind"), ("Archers", "front"), ("Inferno Tower", "pull"), ("Inferno Dragon", "path"), ("Minions", "path"), ("Mega Minion", "path"), ("Tesla", "pull"), ("Electro Wizard", "behind"), ("Witch", "front"), ("Baby Dragon", "path")],
    "Lava Hound": AIR_TANK_KILLERS,
    # spells that land on the tower, answer them the moment they land
    "Goblin Barrel": BARREL_KILLERS,
    "Skeleton Barrel": BARREL_KILLERS,
    "Graveyard": [("Valkyrie", "front"), ("Baby Dragon", "front"), ("Witch", "front"), ("Arrows", "on"), ("The Log", "on"), ("Bomber", "front"), ("Wizard", "front")],
    "Miner": [("Archers", "front"), ("Knight", "front"), ("Mini P.E.K.K.A", "front"), ("Skeletons", "front"), ("Valkyrie", "front"), ("Baby Dragon", "front")],
    "Goblin Drill": [("Valkyrie", "front"), ("Knight", "front"), ("Mini P.E.K.K.A", "front"), ("Arrows", "on"), ("The Log", "on"), ("Baby Dragon", "front")],
    # swarms die to splash and small spells
    "Skeleton Army": SWARM_KILLERS,
    "Goblin Gang": SWARM_KILLERS,
    "Goblins": SWARM_KILLERS,
    "Spear Goblins": SWARM_KILLERS,
    "Barbarians": [("Valkyrie", "path"), ("Baby Dragon", "path"), ("Witch", "behind"), ("Wizard", "behind"), ("Bomber", "behind"), ("Dark Prince", "path"), ("Fireball", "on"), ("Mini P.E.K.K.A", "front")],
    "Guards": SWARM_KILLERS,
    "Royal Recruits": [("Valkyrie", "path"), ("Baby Dragon", "path"), ("Witch", "behind"), ("Bowler", "behind"), ("Dark Prince", "path")],
    "Minions": AIR_SWARM_KILLERS,
    "Minion Horde": AIR_SWARM_KILLERS,
    "Bats": AIR_SWARM_KILLERS,
    "Skeleton Dragons": [("Arrows", "on"), ("Musketeer", "behind"), ("Archers", "behind"), ("Baby Dragon", "path"), ("Wizard", "behind"), ("Witch", "behind")],
    # mini tanks and fighters
    "Knight": MINI_TANK_KILLERS,
    "Valkyrie": [("Mini P.E.K.K.A", "front"), ("Knight", "front"), ("Baby Dragon", "path"), ("Minions", "path"), ("P.E.K.K.A", "front"), ("Cannon", "pull")],
    "Prince": [("Skeleton Army", "front"), ("Skeletons", "front"), ("Guards", "front"), ("Mini P.E.K.K.A", "front"), ("Knight", "front"), ("Cannon", "pull"), ("Barbarians", "front")],
    "Dark Prince": [("Mini P.E.K.K.A", "front"), ("Knight", "front"), ("Cannon", "pull"), ("Barbarians", "front"), ("Minions", "path")],
    "Mini P.E.K.K.A": [("Skeletons", "front"), ("Skeleton Army", "front"), ("Knight", "front"), ("Guards", "front"), ("Cannon", "pull"), ("Archers", "behind"), ("Baby Dragon", "path")],
    "Elite Barbarians": [("Valkyrie", "front"), ("Skeleton Army", "front"), ("Mini P.E.K.K.A", "front"), ("Knight", "front"), ("Barbarians", "front"), ("Witch", "behind")],
    "Lumberjack": [("Knight", "front"), ("Mini P.E.K.K.A", "front"), ("Skeletons", "front"), ("Archers", "behind"), ("Cannon", "pull")],
    "Bandit": [("Knight", "front"), ("Mini P.E.K.K.A", "front"), ("Skeletons", "front"), ("Valkyrie", "front"), ("Cannon", "pull")],
    "Royal Ghost": [("Knight", "front"), ("Valkyrie", "front"), ("Mini P.E.K.K.A", "front"), ("Skeletons", "front"), ("Arrows", "on")],
    "Sparky": [("Zap", "on"), ("Electro Wizard", "path"), ("Skeleton Army", "front"), ("Mini P.E.K.K.A", "front"), ("Knight", "front"), ("Barbarians", "front"), ("Minions", "path")],
    # ranged and support, get on top of them fast
    "Musketeer": RANGED_KILLERS,
    "Wizard": RANGED_KILLERS,
    "Witch": RANGED_KILLERS + [("Valkyrie", "path")],
    "Executioner": RANGED_KILLERS,
    "Electro Wizard": RANGED_KILLERS,
    "Night Witch": [("Valkyrie", "path"), ("Mini P.E.K.K.A", "path"), ("Knight", "path"), ("Baby Dragon", "path"), ("Fireball", "on")],
    "Princess": [("The Log", "on"), ("Arrows", "on"), ("Zap", "on"), ("Barbarian Barrel", "on"), ("Fireball", "on")],
    "Dart Goblin": [("The Log", "on"), ("Arrows", "on"), ("Zap", "on"), ("Fireball", "on"), ("Knight", "path"), ("Mini P.E.K.K.A", "path")],
    "Archers": [("Arrows", "on"), ("The Log", "on"), ("Fireball", "on"), ("Valkyrie", "path"), ("Knight", "path"), ("Mini P.E.K.K.A", "path"), ("Baby Dragon", "path")],
    "Baby Dragon": [("Musketeer", "behind"), ("Archers", "behind"), ("Mega Minion", "path"), ("Electro Wizard", "behind"), ("Witch", "behind"), ("Minions", "path"), ("Inferno Dragon", "path")],
    "Inferno Dragon": [("Zap", "on"), ("Electro Spirit", "path"), ("Musketeer", "behind"), ("Archers", "behind"), ("Electro Wizard", "behind"), ("Minions", "path"), ("Arrows", "on")],
    "Mega Minion": [("Archers", "behind"), ("Musketeer", "behind"), ("Arrows", "on"), ("Baby Dragon", "path"), ("Witch", "behind")],
    "X-Bow": [("Giant", "on"), ("Knight", "on"), ("Valkyrie", "on"), ("Mini P.E.K.K.A", "on"), ("P.E.K.K.A", "on"), ("Fireball", "on")],
    "Mortar": [("Knight", "on"), ("Mini P.E.K.K.A", "on"), ("Valkyrie", "on"), ("Giant", "on"), ("Miner", "on")],
}


def answers_for(threat: str) -> list[tuple[str, str]]:
    return ANSWERS.get(threat, [])


def spot(kind: str, lane: str, threat: str) -> str:
    return SPOTS[kind].format(lane=lane, threat=threat)


def spot_xy(kind: str, lane: str, threat_at: tuple[int, int] | None = None) -> tuple[int, int]:
    """Where on the screen a spot is, in canonical pixels."""
    col = LANE_COL[lane]
    if kind == "on" and threat_at:
        return threat_at
    if kind == "path" and threat_at:
        # a few tiles ahead of the threat on its way down the lane
        return tile(col, max(1, min(14, row_of(threat_at[1]) - 3)))
    return {
        "pull": tile(8.5, 11),
        "front": tile(col, 10),
        "behind": tile(col, 3),
        "bridge": tile(col, 13),
        "back": tile(6 if lane == "left" else 11, 0),
        "path": tile(col, 11),
        "on": tile(col, 10),
    }[kind]
