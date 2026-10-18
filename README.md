<h1 align="center">Clash Helper</h1>

<p align="center">A live coach for Clash Royale that sits beside your phone.</p>

<p align="center">
  <img src="docs/demo.gif" alt="Clash Helper calling plays next to a match" width="720">
</p>

Clash Helper watches your match through iPhone Mirroring on a Mac. It works out which cards your opponent is holding, how much elixir they have, and tells you what to play next, where to put it, and why.

## What it shows

- **One call at a time.** `Hog Rider, right bridge: no answer in their hand`. Urgent defense calls turn red.
- **Their hand.** Clash Royale decks cycle in a fixed order, so once their cards are known the helper knows exactly which 4 they hold and which one comes next.
- **Their elixir.** Rebuilt from the match clock and every card they play.
- **A deck for you.** Picked from thousands of real battles near your trophy count, limited to decks you can build at good levels.

## Setup

Requires macOS with iPhone Mirroring and Python 3.11+.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
```

1. Create a key at [developer.clashroyale.com](https://developer.clashroyale.com) with the IP `45.79.218.79`. That is the [RoyaleAPI relay](https://docs.royaleapi.com/proxy.html), so the key keeps working when your own IP changes. Put it in `.env` as `CR_API_TOKEN`.
2. Put your player tag (on your profile, without the `#`) in `.env` as `CR_PLAYER_TAG`.
3. Run `clash-helper setup` once. It installs the [KataCR](https://github.com/wty-yy/KataCR) unit detector in its own environment under `~/Library/Caches/clash-helper` (a few GB with PyTorch).
4. Give your terminal **Screen Recording** and **Accessibility** access in System Settings > Privacy & Security.

## Use

```bash
clash-helper                         # coach the match on your phone
clash-helper deck                    # suggest a deck from your cards
clash-helper replay match.mp4        # print the calls for a screen recording
clash-helper replay match.mp4 --panel
clash-helper record                  # save frames of a match for testing
```

Open iPhone Mirroring first. The helper snaps it to the right half of the screen and opens its panel on the left.

## How it works

1. Each frame of the iPhone Mirroring window is cropped to one fixed screen layout.
2. The KataCR detector, running on the Mac's GPU in the background, sees every unit, spell, tower and deploy timer on the arena and follows each one across frames.
3. Enemy cards land with a red stopwatch, and the units that appear on it name the card. Spells are told apart from yours by the way they travel, a Zap and an Electro Wizard by the bolt they drop from the sky, Arrows by the ring they leave.
4. Their deck comes from the official API when they are in a clan. Most opponents near 4,000 trophies are ladder bots that cannot be looked up, but they play preset decks, so 3 or 4 of their cards give away the other 4.
5. Because no card can repeat within 4 plays, the whole sequence of their plays is named at once, so later plays correct earlier mistakes. Their elixir is charged the moment a card lands.
6. A small set of rules picks the call: defend, punish low elixir, attack while their answer is out of hand, avoid sitting at full elixir, or wait.

See [docs/design.md](docs/design.md) for the details.

## Limits

- On 13 hand labeled matches, each scored with what was learned from the other 12: 92% of enemy plays caught and 88% named right with their deck known, 90% and 84% the way the helper really runs (deck looked up or guessed). Their elixir is off by 1.2 to 1.7 on average, and a play shows up about 1.2 s after the card leaves their hand.
- The detector was trained in 2024. It never sees Vines, rarely sees a Rocket in flight, and mixes up some newer evolved skins until it is retrained on recent matches.
- 1v1 trophy matches only.

## Development

```bash
.venv/bin/pytest
python scripts/build_cards.py        # refresh card costs and art from the API
python scripts/demo.py match.mp4     # rebuild docs/demo.gif
python research/report.py            # score every hand labeled match in captures/, see docs/design.md
```

Unit detection comes from [KataCR](https://github.com/wty-yy/KataCR) (MIT), installed by `clash-helper setup` rather than shipped here.

Clash Helper is a fan project and is not affiliated with or endorsed by Supercell.
