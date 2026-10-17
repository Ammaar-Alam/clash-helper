"""Write src/clash_helper/data/cards.json with every card's name, cost, type and icon.

Uses the official API when CR_API_TOKEN is set, otherwise RoyaleAPI's open card list.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from clash_helper import api

OUT = Path(__file__).resolve().parents[1] / "src/clash_helper/data/cards.json"
OPEN_LIST = "https://raw.githubusercontent.com/RoyaleAPI/cr-api-data/master/docs/json/cards.json"
# official ids start with the card type
TYPES = {"26": "troop", "27": "building", "28": "spell"}


def main() -> None:
    if api.token():
        cards = [
            {
                "name": c["name"],
                "cost": c.get("elixirCost"),
                "type": TYPES.get(str(c["id"])[:2], "troop"),
                "rarity": c["rarity"],
                "icon": c["iconUrls"].get("medium"),
                "evo_icon": c["iconUrls"].get("evolutionMedium"),
            }
            for c in api.cards()
        ]
    else:
        with urllib.request.urlopen(OPEN_LIST, timeout=10) as response:
            data = json.load(response)
        cards = [
            {"name": c["name"], "cost": c["elixir"], "type": c["type"].lower(), "rarity": c["rarity"], "icon": None, "evo_icon": None}
            for c in data
        ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sorted(cards, key=lambda c: c["name"]), indent=1) + "\n")
    print(f"wrote {len(cards)} cards to {OUT}")


if __name__ == "__main__":
    main()
