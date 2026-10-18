import json
from collections import Counter

from clash_helper import bots, confusion, kcr
from clash_helper.decode import Evidence, expected_cost
from clash_helper.game import Match, Pending


class Idle:
    """A detector that never sees anything, the tests feed units straight in."""

    busy = False

    def submit(self, frame, now):
        return True

    def result(self):
        return 0.0, []


def test_a_rolling_log_split_over_track_ids_is_one_cast():
    m = Match(detector=Idle(), lookup=False, sync=True)
    m.started = 0.0

    def log(y: int, track: int) -> kcr.Unit:
        # the detector calls it yours, it rolls down the screen so it is theirs
        return kcr.Unit("the-log", "The Log", False, 0.93, 160, y - 20, y, track)

    m._see([log(506, 1)], 80.0)
    m._see([log(590, 2)], 80.33)  # the tracker lost it and gave it a new id
    m._see([log(663, 3)], 81.17)
    assert [p.time for p, e in m.events if e.spell] == [80.0]


def test_a_bolt_covers_its_whole_column():
    bolt = Pending(150, 500, 10.0, 1.0, source="bolt")
    timer = Pending(150, 500, 10.0, 1.0)
    assert bolt.covers(160, 900) and not timer.covers(160, 900)


def test_bot_deck_needs_three_cards_and_forgives_one_stray(monkeypatch, tmp_path):
    pool = [[list("ABCDEFGH"), 5], [list("ABCDEFGX"), 1], [list("ABQRSTUV"), 3]]
    path = tmp_path / "bots.json"
    path.write_text(json.dumps(pool))
    monkeypatch.setenv("CLASH_BOT_DECKS", str(path))
    bots.decks.cache_clear()
    try:
        assert bots.deck(frozenset("AB")) is None
        assert bots.deck(frozenset("ABC")) == list("ABCDEFGH")
        assert bots.deck(frozenset("ABCDZ")) == list("ABCDEFGH")
        assert bots.deck(frozenset("ABCYZ")) is None
    finally:
        bots.decks.cache_clear()


def test_an_unsure_name_costs_between_its_readings(monkeypatch):
    monkeypatch.setattr(confusion, "table", lambda: {"Prince": {"Prince": 1, "Three Musketeers": 1}})
    evidence = Evidence(Counter({"Prince": 1.0}), 1, None)
    assert 5 < expected_cost(evidence, ["Prince", "Three Musketeers"], {}) < 9
