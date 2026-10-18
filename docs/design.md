# Design

A live coach for 1v1 trophy matches. It watches the iPhone Mirroring window on a Mac, tracks the opponent's cards and elixir, and shows one call at a time next to the phone: what to play, where, and why, with the plan for what comes next.

## Goals

- One glanceable call at a time with a short reason, the spot drawn on the phone and said in words.
- Catch every enemy play quickly, spells included, so their elixir and hand are right.
- Work with any deck, the player's own deck is read from the API at the start of every match.
- Record every match so accuracy can be measured and improved on real games.

## Non-goals

- Playing for the user. The helper never clicks or touches the phone.
- 2v2, draft, events and other modes.

## Pipeline

```
iPhone Mirroring window
  -> capture.py    grab the window by id at full resolution, even when covered
  -> screen.py     crop to one canonical 664x1440 screen so every region is fixed
  -> game.observe  what one frame shows: match clock, elixir, hand, deploy timers, marks, units
  -> game.update   tracks, landings, spells, names, towers, threat
  -> tracker.py    their card cycle and elixir
  -> advisor.py    one call: card, spot, reason, plan
  -> panel.py, overlay.py, journal.py
```

`observe` is the only part that needs the image. Recordings are replayed through `update` from cached observations, so experiments run in seconds and live play and replays share one code path.

### Units and spells

The [KataCR](https://github.com/wty-yy/KataCR) detector runs in its own environment as a background worker on the GPU, about 220 ms a frame. It reports every unit, spell, tower and deploy timer with a side and a track id that follows it across frames. The tracker is not perfect, so tracks are repaired before use: a spell it lost and found again under a new id is joined back to its first sighting, a spell that inherits a troop's id (or the other way round) is split off, and a unit that takes over another unit's id votes only with the classes that are new on that track.

### Enemy plays

- **Deploy timers.** An enemy card lands with a red stopwatch. The detector spots it about half a second before a plain red disc check does, but now and then mistakes one of the player's timers for theirs, so its timers count only on their half. Each timer opens a landing.
- **New units.** A confident new enemy unit on their half with no timer nearby also opens a landing, for cards placed behind their king tower and cards without a timer. Units near a unit of the same card, spawns of a card already on the arena, and units on the player's half are not new plays.
- **Spells.** Spells roll and fly from their side down toward the player's and the player's own spells go up, which tells whose a spell is better than the detector's side guess. A spell seen only where it lands, such as a Fireball, needs one confident detection and is skipped when the player just cast that card. Effects of a unit landing (the Electro Wizard's zap, the Ice Wizard's burst) are not spells.
- **Bolts.** A Zap and an Electro Wizard both drop in as a bolt through the strip at the top of the screen. A unit appearing under the bolt makes it an Electro Wizard, nothing makes it a Zap.
- **Arrows.** Enemy Arrows leave a salmon ring on the ground.

Each landing waits 1.4 seconds while the units that appeared on it vote on the card. Their elixir is charged the moment the landing opens, at the average cost of their hand, and corrected once the card is named.

### Naming

The detector was trained on older cards and confuses look-alikes. `data/confusion.json`, learned from hand labeled recordings, says what each detector class really turned out to be, including no play at all. Each candidate also carries how many units landed at once, since a swarm cannot hide behind a single unit's name. With their deck known, the whole sequence of plays is decoded at once with a beam search under the cycle rule (no card repeats within 4 plays), so later plays fix earlier names.

### Their deck

1. **Lookup.** Their name, clan and trophies are read off the screen and the API finds the player in that clan. The trophies must agree, so a namesake's deck is never used.
2. **Bot decks.** Most opponents near 4300 trophies are ladder bots. They show a clan name the API does not have, but they play a fixed set of preset decks, gathered from battle logs by `research/bots.py`. Once 3 or 4 of their cards are named and one preset holds them, that deck is used as if looked up. It is worked out again on every play and dropped when their cards stop fitting it.
3. **From play.** Otherwise the 8 cards their plays most strongly point to.

### Card cycle

Decks hold 8 cards: 4 in hand, 4 in a queue. A played card goes to the back of the queue and the front card enters the hand, so their hand is their deck minus the last 4 plays and the next card is the one played 4 plays ago. Champions leave the cycle while alive.

### Elixir

Both players start at 5 with a cap of 10 and gain 1 every 2.8 seconds, twice as fast once the double elixir badge shows. The helper often notices a match a few seconds in, so the match clock sets the start. A play costing more than the estimate proves the estimate was low, so a missed play never drags it below zero.

### Towers

A princess tower the detector has not seen for 4 seconds is destroyed. Health numbers are read with text recognition inside the detector's health bar box, since tower troops differ in height.

### Advice

Calls come from card facts (cost, targets, splash, swarm, air, building, spell size) and a playbook of answers and spots, not a trained model. In priority order: answer a fresh push, defend units already on your side, spell a cluster, support your tank, counterpush with survivors, punish when they spent big, finish a tower one spell from falling, attack while their answer is out of hand, build a push from the back, cycle toward your attacker, do not sit at full elixir, and otherwise say what you are saving for.

## Panel and overlay

The panel sits to the left of the phone: the call in large type, their hand and next card, both elixir counts, a recording toggle and an info button. A see-through overlay above the phone draws the card at its spot and ignores the mouse, so dragging works straight through it.

## Recording and measurement

Every match is saved under `~/Library/Application Support/clash-helper/games` with the video, a line per event and a summary that picks up the result and both true decks from the battle log afterwards.

Recordings are labeled by hand, every opponent play with its first frame, place and card. The research loop:

1. `scripts/scan.py` caches the detector's output per recording, `research/observe.py` everything else a frame shows.
2. `research/events.py` pairs every candidate play with the labels, `research/train.py` learns the class meanings.
3. `research/report.py` scores each recording with class meanings learned from the other recordings only, with their deck known and learned from play: plays found, false plays, plays named right, spells, how far the elixir estimate is from the labels frame by frame, and how long a play takes to show. `research/elixir_breakdown.py` splits the elixir error into delay, names, missed and false plays.

## Testing

Pure logic (cycle, elixir, advice, tracking) has unit tests. Recordings stay out of git.
