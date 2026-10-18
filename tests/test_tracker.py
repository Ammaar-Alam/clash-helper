from clash_helper.tracker import Opponent

DECK = ["hog", "musketeer", "fireball", "log", "ice spirit", "skeletons", "cannon", "ice golem"]


def play_all(opp: Opponent, cards: list[str], start: float = 0.0) -> None:
    for i, card in enumerate(cards):
        opp.play(card, 1, now=start + i)


def test_hand_unknown_until_cards_cycle_back():
    opp = Opponent()
    play_all(opp, DECK[:3])
    assert opp.hand == []
    assert opp.next_card is None


def test_full_deck_gives_exact_hand_and_next_card():
    opp = Opponent()
    play_all(opp, DECK)
    assert opp.hand == DECK[:4]
    assert opp.next_card == "ice spirit"
    opp.play("hog", 4, now=20)
    assert opp.hand == ["musketeer", "fireball", "log", "ice spirit"]
    assert opp.next_card == "skeletons"
    assert opp.plays_until_back("hog") == 4
    assert opp.plays_until_back("cannon") == 2


def test_card_seen_outside_last_four_is_in_hand_before_deck_complete():
    opp = Opponent()
    play_all(opp, DECK[:5])
    assert opp.hand == ["hog"]


def test_champion_leaves_cycle_until_it_dies():
    opp = Opponent()
    play_all(opp, DECK)
    opp.play("golden knight", 4, now=20, champion=True)
    # front of the queue refilled the hand, queue is one shorter
    assert opp.queue == ["skeletons", "cannon", "ice golem"]
    assert "ice spirit" in opp.hand
    opp.champion_died("golden knight")
    assert opp.queue[-1] == "golden knight"
    assert "golden knight" not in opp.hand


def test_elixir_regenerates_and_caps():
    opp = Opponent()
    opp.tick(2.8)
    assert abs(opp.elixir - 6) < 1e-9
    opp.tick(100)
    assert opp.elixir == 10
    opp.play("golem", 8, now=100)
    assert abs(opp.elixir - 2) < 1e-9
    opp.tick(101.4, multiplier=2)
    assert abs(opp.elixir - 3) < 1e-9


def test_missed_play_never_leaves_estimate_below_a_seen_cost():
    opp = Opponent()
    opp.play("golem", 8, now=0)
    opp.play("pekka", 7, now=1)
    assert opp.elixir == 0


def test_preloaded_deck_gives_exact_hand_after_four_plays():
    opp = Opponent()
    opp.preload(DECK)
    play_all(opp, ["hog", "log", "cannon", "skeletons"])
    assert sorted(opp.hand) == sorted(["musketeer", "fireball", "ice spirit", "ice golem"])
    assert opp.next_card == "hog"


def test_preloaded_deck_before_four_plays_only_lists_maybes():
    opp = Opponent()
    opp.preload(DECK)
    play_all(opp, ["hog", "log"])
    assert opp.hand == []
    assert "hog" not in opp.maybe and len(opp.maybe) == 6


def test_unknown_play_keeps_hand_uncertain():
    opp = Opponent()
    play_all(opp, DECK)
    opp.play("?", 3, now=20)
    assert opp.hand == []
    assert sorted(opp.maybe) == sorted(DECK[:5])
    assert "?" not in opp.deck


def test_preload_overrules_wrong_guesses():
    opp = Opponent()
    play_all(opp, ["hog", "giant"])
    opp.preload(DECK)
    assert opp.plays == ["hog", "?"]
    assert opp.deck == DECK


def test_card_back_too_early_means_missed_plays():
    opp = Opponent()
    play_all(opp, DECK)
    # cannon is second in the queue, so two plays went unseen
    opp.play("cannon", 3, now=20)
    assert opp.plays[-3:] == ["?", "?", "cannon"]
    assert opp.hand == []


def test_decoder_keeps_spells_for_spell_evidence():
    from collections import Counter

    from clash_helper.decode import Evidence, decode

    deck = ["Skeleton Army", "Mega Knight", "Princess", "Royal Ghost", "Vines", "Bomber", "Goblin Barrel", "Giant Snowball"]
    evidence = [
        Evidence(Counter({"Skeleton Army": 2.0}), 12, None),
        Evidence(Counter({"Giant Snowball": 2.0}), 0, None, spell=True, source="spell"),
        Evidence(Counter({"Bomber": 1.0}), 1, None),
    ]
    assert decode(evidence, deck, {}) == ["Skeleton Army", "Giant Snowball", "Bomber"]


def test_decoder_drops_an_effect_it_has_learned_is_no_play(monkeypatch):
    from collections import Counter

    from clash_helper import confusion
    from clash_helper.decode import NONE, Evidence, decode

    # pretend the labels showed "Rage" sightings were never real plays
    monkeypatch.setattr(confusion, "table", lambda: {"Rage": {"none": 20}})
    deck = ["Witch", "Minions", "Dart Goblin", "Fireball", "Goblin Gang", "Skeleton Barrel", "Electro Wizard", "Mega Knight"]
    evidence = [Evidence(Counter({"Rage": 1.0}), 0, None, spell=True, source="spell")]
    assert decode(evidence, deck, {}) == [NONE]
