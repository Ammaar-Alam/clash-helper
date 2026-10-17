from clash_helper.advisor import MyPlay, Situation, Threat, advise
from clash_helper.board import Board
from clash_helper.cards import card
from clash_helper.tracker import Opponent

HAND = [card(n) for n in ("Hog Rider", "Valkyrie", "Musketeer", "The Log")]
BEATDOWN = [card(n) for n in ("Giant", "Witch", "Archers", "Arrows")]


def opponent(deck: list[str], elixir: float = 5) -> Opponent:
    opp = Opponent()
    for i, name in enumerate(deck):
        opp.play(name, 0, now=i)
    opp.elixir = elixir
    return opp


THEIRS = opponent(["Golem", "Baby Dragon", "Lightning", "Mega Minion", "Night Witch", "Zap", "Lumberjack", "Tornado"], 7)


def call(hand=HAND, elixir=5, opp=None, **kw):
    return advise(Situation(now=kw.pop("now", 10), hand=hand, elixir=elixir, opp=opp or Opponent(), **kw))


def test_defends_swarm_with_the_log_on_it():
    c = call(threat=Threat("Skeleton Army", 500, 9, 600), answered=False)
    assert c.card == "The Log" and c.urgent and c.target is not None


def test_air_threat_gets_an_air_answer():
    c = call(threat=Threat("Baby Dragon", 100, 9, 600), answered=False)
    assert c.card == "Musketeer" and c.action == "Musketeer, behind your left tower"


def test_gets_ready_when_answer_is_unaffordable():
    c = call(elixir=2, threat=Threat("Baby Dragon", 100, 9, 600), answered=False)
    assert c.rule == "save" and c.card is None and "Musketeer" in c.plan


def test_cheap_unit_is_left_to_the_tower():
    assert not call(threat=Threat("Fire Spirit", 500, 9, 600), answered=False).urgent


def test_playbook_pulls_hog_with_a_building():
    hand = [card(n) for n in ("Cannon", "Mini P.E.K.K.A", "Arrows", "Giant")]
    c = call(hand=hand, threat=Threat("Hog Rider", 500, 9, 600), answered=False)
    assert c.action == "Cannon, center, 3 tiles from the river"


def test_punishes_low_elixir_in_the_other_lane():
    opp = Opponent()
    opp.play("Golem", 8, now=1)
    c = call(now=2, elixir=6, opp=opp, threat=Threat("Golem", 100, 1))
    assert c.action == "Hog Rider, right bridge" and c.rule == "punish"


def test_support_goes_behind_your_tank():
    c = call(hand=[card(n) for n in ("Witch", "Archers", "Arrows", "Cannon")], elixir=6, opp=THEIRS, now=14,
             mine=MyPlay("Giant", 10, "left"))
    assert c.card == "Witch" and c.rule == "support"


def test_beatdown_starts_a_push_from_the_back():
    c = call(hand=BEATDOWN, elixir=8, opp=THEIRS)
    assert c.card == "Giant" and c.rule == "push" and "king tower" in c.action


def test_spell_hits_a_bunched_group():
    c = call(opp=THEIRS, enemies=[(300, 660), (320, 670), (310, 690), (600, 300)])
    assert c.card == "The Log" and c.rule == "spell"


def test_finish_a_low_tower_with_a_spell():
    board = Board()
    board.their_towers = {"left": 3052, "right": 60}
    assert call(opp=THEIRS, board=board).card == "The Log"


def test_units_across_the_river_need_defending():
    board = Board()
    board.enemy_home = {"left": 3, "right": 0}
    c = call(opp=THEIRS, board=board, answered=False)
    assert c.urgent and c.card == "Valkyrie"


def test_waiting_always_has_a_plan():
    c = call(hand=BEATDOWN, elixir=3, opp=THEIRS)
    assert c.rule == "wait" and c.plan.startswith("Giant")


def test_cycle_toward_a_missing_win_card():
    hand = [card(n) for n in ("Valkyrie", "Musketeer", "The Log", "Ice Spirit")]
    c = call(hand=hand, elixir=7, opp=THEIRS)
    assert c.rule == "cycle" and c.card == "Ice Spirit"


def test_a_hog_already_at_your_tower_gets_a_troop_on_it_not_a_building():
    hand = [card(n) for n in ("Cannon", "Mini P.E.K.K.A", "Arrows", "Giant")]
    c = call(hand=hand, threat=Threat("Hog Rider", 520, 9, 860), answered=False)
    assert c.card == "Mini P.E.K.K.A" and "right on the Hog Rider" in c.action
    assert c.target[1] > 860


def test_a_tank_on_their_side_is_not_a_defense_call_yet():
    c = call(hand=BEATDOWN, threat=Threat("Golem", 300, 9, 200), answered=False)
    assert c.rule != "defend"
