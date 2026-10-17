from clash_helper.deck import rank

GOOD = ["Hog Rider", "Musketeer", "Fireball", "The Log", "Ice Spirit", "Skeletons", "Cannon", "Ice Golem"]
OTHER = ["Giant", "Witch", "Arrows", "Zap", "Knight", "Archers", "Minions", "Valkyrie"]


def side(tag: str, cards: list[str], crowns: int) -> list[dict]:
    return [{"tag": tag, "crowns": crowns, "cards": [{"name": c} for c in cards]}]


def battle(n: int, winner: str, deck: list[str], won: bool) -> dict:
    return {
        "battleTime": str(n),
        "team": side(winner, deck, 1 if won else 0),
        "opponent": side(f"#X{n}", OTHER + ["Bats"], 0 if won else 1),
    }


def player(level: int = 11) -> dict:
    cards = [{"name": c, "level": level, "maxLevel": 16} for c in GOOD + OTHER]
    return {"tag": "#ME", "cards": cards}


def test_deck_needs_several_players():
    battles = [battle(i, "#A", GOOD, True) for i in range(10)]
    assert rank(player(), battles) == []


def test_one_player_cannot_flood_the_record():
    battles = [battle(i, "#A", GOOD, True) for i in range(10)] + [battle(10 + i, f"#B{i}", GOOD, False) for i in range(2)]
    pick = rank(player(), battles)[0]
    assert pick.games == 7 and pick.wins == 5 and pick.players == 3


def test_own_battles_are_ignored():
    battles = [battle(i, "#ME", GOOD, True) for i in range(5)]
    assert rank(player(), battles) == []
