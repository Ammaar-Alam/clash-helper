from collections import Counter

from clash_helper.identify import name_deploy

DECK = ["Archers", "Knight", "Mega Knight", "Fire Spirit", "Dart Goblin", "Skeletons", "Musketeer", "Electro Spirit"]


def test_votes_pick_the_card_whose_main_unit_was_seen():
    assert name_deploy(Counter({"dart_goblin": 3.0, "goblin": 0.5}), DECK, 1) == "Dart Goblin"


def test_badge_count_names_a_swarm_the_detector_missed():
    assert name_deploy(Counter(), DECK, 2) == "Archers"
    assert name_deploy(Counter(), DECK, 3) == "Skeletons"


def test_nothing_seen_is_unknown():
    assert name_deploy(Counter(), DECK, 1) is None


def test_unlikely_cards_lose_close_calls():
    votes = Counter({"musketeer": 1.0, "archer": 0.8})
    assert name_deploy(votes, DECK, 1, unlikely={"Musketeer"}) == "Archers"


def test_without_a_known_deck_skeletons_and_army_split_by_count():
    assert name_deploy(Counter({"skeleton": 2.0}), None, 3) == "Skeletons"
    assert name_deploy(Counter({"skeleton": 2.0}), None, 14) == "Skeleton Army"
