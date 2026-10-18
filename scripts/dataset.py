"""Collect every recording with its true decks into captures/dataset.json.

    python scripts/dataset.py

Screen recordings live in captures/, the helper's own recordings in the games folder.
True decks come from the battle log, kept in captures/matches.json or the game summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from clash_helper.journal import GAMES

CAPTURES = Path(__file__).resolve().parents[1] / "captures"


def video_info(path: Path) -> dict:
    video = cv2.VideoCapture(str(path))
    info = {
        "fps": round(video.get(cv2.CAP_PROP_FPS), 2),
        "width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "seconds": round(video.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, video.get(cv2.CAP_PROP_FPS)), 1),
    }
    video.release()
    return info


def main() -> None:
    items = []
    known = json.loads((CAPTURES / "matches.json").read_text())
    for name, meta in sorted(known.items()):
        path = CAPTURES / f"{name}.mp4"
        if path.exists():
            labels = CAPTURES / f"{name}_labels.json"
            items.append({"id": name, "video": str(path), "opponent": meta["opponent"], "their_deck": meta["deck"],
                          "my_deck": meta.get("mine", []), "labels": str(labels) if labels.exists() else None,
                          **video_info(path)})
    for folder in sorted(GAMES.glob("*")):
        summary, video = folder / "summary.json", folder / "match.mp4"
        if not (summary.exists() and video.exists()):
            continue
        s = json.loads(summary.read_text())
        result = s.get("result") or {}
        info = video_info(video)
        if not result or info["seconds"] < 20:
            continue  # unfinished or broken recording
        labels = CAPTURES / f"game_{folder.name}_labels.json"
        items.append({"id": f"game_{folder.name}", "video": str(video), "opponent": result.get("opponent"),
                      "their_deck": result.get("their_deck", []), "my_deck": result.get("my_deck", []),
                      "won": result.get("won"), "labels": str(labels) if labels.exists() else None, **info})
    (CAPTURES / "dataset.json").write_text(json.dumps(items, indent=1))
    for item in items:
        print(f"{item['id']:<28} {item['width']}x{item['height']} {item['fps']:>5} fps {item['seconds']:>6}s "
              f"labels {'yes' if item['labels'] else 'no ':<3} vs {item['opponent']}")


if __name__ == "__main__":
    main()
