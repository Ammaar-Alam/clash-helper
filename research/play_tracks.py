"""Which detector tracks each labeled enemy play put on the arena, to correct the detector's labels.

    python research/play_tracks.py

Replays every labeled recording with their deck known, pairs the helper's plays with the labels and
keeps, for each labeled play the helper found, the raw tracker ids of the units that voted for it, or
of the spell that showed it. Writes captures/play_tracks.json: recording -> [{card, spell, time, tracks}].
research/detector_data.py relabels those tracks' boxes with the true card.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import CAPTURES, CachedDetector, observations, pair, truth_plays  # noqa: E402

from clash_helper.game import Match  # noqa: E402


def tracks_of(item: dict) -> list[dict]:
    raw: dict[int, set[int]] = {}  # track key -> raw tracker ids that fed it
    original = Match._key

    def key(self, u, at, present):
        k = original(self, u, at, present)
        raw.setdefault(k, set()).add(u.track)
        return k

    Match._key = key
    try:
        match = Match(detector=CachedDetector([[0, []]]), lookup=False, sync=True)
        events: list = []
        for obs in observations(item, match, False):
            match.update(obs)
            if match.started is not None and not match.deck_from_api:
                match.opp.preload(item["their_deck"])
                match.deck_from_api = True
            if match.events:
                events = list(match.events)
    finally:
        Match._key = original
    truth = truth_plays(json.loads(Path(item["labels"]).read_text()))
    hits = pair(truth, [(p.time, p.x, p.y, e.spell) for p, e in events])
    out = []
    for (p, e), h in zip(events, hits):
        if h is None:
            continue
        keys = {e.track} if e.spell and e.track is not None else set(p.tracks)
        ids = sorted({i for k in keys for i in raw.get(k, {k})})
        if ids:
            out.append({"card": truth[h]["card"], "spell": e.spell, "time": truth[h]["time"], "tracks": ids})
    return out


def main() -> None:
    items = [i for i in json.loads((CAPTURES / "dataset.json").read_text()) if i["labels"]]
    result = {}
    for item in items:
        result[item["id"]] = tracks_of(item)
        print(f"{item['id']:<28} {len(result[item['id']])} plays with tracks", flush=True)
    (CAPTURES / "play_tracks.json").write_text(json.dumps(result, indent=0))


if __name__ == "__main__":
    main()
