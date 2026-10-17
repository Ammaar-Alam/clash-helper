from clash_helper.advisor import Threat, advise
from clash_helper.cards import card
from clash_helper.tracker import Opponent

HAND = [card(n) for n in ("Hog Rider", "Valkyrie", "Musketeer", "The Log")]
THEIR_DECK = ["Golem", "Baby Dragon", "Lightning", "Mega Minion", "Night Witch", "Zap", "Lumberjack", "Tornado"]


def full_opponent(elixir: float = 5) -> Opponent:
    opp = Opponent()
    for i, name in enumerate(THEIR_DECK):
        opp.play(name, 0, now=i)
    opp.elixir = elixir
    return opp


def test_defends_swarm_with_cheapest_answer():
    call = advise(10, HAND, 5, Opponent(), Threat("Skeleton Army", 500, 9), answered=False)
    assert call.action.startswith("The Log")
    assert call.urgent


def test_air_threat_needs_air_answer():
    call = advise(10, HAND, 5, Opponent(), Threat("Baby Dragon", 100, 9), answered=False)
    assert call.action.startswith("Musketeer, in front of left tower")


def test_saves_when_answer_is_unaffordable():
    call = advise(10, HAND, 2, Opponent(), Threat("Baby Dragon", 100, 9), answered=False)
    assert call.action == "Save for Musketeer"


def test_punishes_low_elixir_in_the_other_lane():
    opp = Opponent()
    opp.play("Golem", 8, now=1)
    call = advise(2, HAND, 6, opp, Threat("Golem", 100, 1), answered=True)
    assert call.action == "Hog Rider, right bridge"
    assert "elixir" in call.reason


def test_safe_window_when_no_counter_in_hand():
    opp = Opponent()
    for i, name in enumerate(["Golem", "Lightning", "Zap", "Tornado", "Baby Dragon", "Mega Minion", "Night Witch", "Lumberjack"]):
        opp.play(name, 0, now=i)
    # their hand is Golem Lightning Zap Tornado and nothing affordable stops a Hog
    opp.elixir = 5
    call = advise(20, HAND, 5, opp, None, answered=True)
    assert call.reason == "no answer in their hand"


def test_does_not_leak_at_full_elixir():
    call = advise(20, HAND, 10, full_opponent(9), None, answered=True)
    assert "full" in call.reason


def test_unknown_hand_still_gives_timing():
    opp = Opponent()
    opp.play("Golem", 8, now=1)
    call = advise(2, [None] * 4, 7, opp, None, answered=True)
    assert call.action == "Attack right"
