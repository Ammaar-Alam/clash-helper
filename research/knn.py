"""How well DINOv2 crops alone name plays, by nearest labeled neighbours.

Cross game: references come only from other recordings (leave one recording out).
Same game: references are the earlier plays of the same recording, what the helper
would know mid match. Both restricted to the opponent's true deck.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

CAPTURES = Path(__file__).resolve().parents[1] / "captures"


def load() -> dict:
    data = {}
    decks = {i["id"]: i["their_deck"] for i in json.loads((CAPTURES / "dataset.json").read_text())}
    for path in sorted((CAPTURES / "crops").glob("*.npz")):
        emb = path.with_suffix(".emb.npy")
        if emb.exists():
            z = np.load(path)
            e = np.load(emb).mean(axis=1)
            data[path.stem] = (e / np.linalg.norm(e, axis=1, keepdims=True), list(z["cards"]), decks[path.stem])
    return data


def vote(query: np.ndarray, refs: np.ndarray, labels: list[str], allowed: set[str], k: int = 5) -> str | None:
    keep = [i for i, c in enumerate(labels) if c in allowed]
    if not keep:
        return None
    sims = refs[keep] @ query
    order = np.argsort(-sims)[:k]
    score = Counter()
    for i in order:
        score[labels[keep[i]]] += float(sims[i])
    return score.most_common(1)[0][0]


def main() -> None:
    data = load()
    for rid, (emb, cards, deck) in data.items():
        others = [(e, c) for r, (e, c, _) in data.items() if r != rid]
        refs = np.concatenate([e for e, _ in others]) if others else np.zeros((0, emb.shape[1]))
        labels = [c for _, cs in others for c in cs]
        covered = sum(c in labels for c in cards)
        cross = sum(vote(emb[i], refs, labels, set(deck)) == c for i, c in enumerate(cards) if c in labels)
        same = 0
        tried = 0
        for i, c in enumerate(cards):
            earlier = list(range(i))
            if c in [cards[j] for j in earlier]:
                tried += 1
                same += vote(emb[i], emb[earlier], [cards[j] for j in earlier], set(deck), k=3) == c
        print(f"{rid:<28} cross game {cross}/{covered} (of {len(cards)} plays, rest unseen elsewhere) | same game repeats {same}/{tried}")


if __name__ == "__main__":
    main()
