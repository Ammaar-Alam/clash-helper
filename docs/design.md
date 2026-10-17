# Design

A live coach for 1v1 trophy matches. It watches the iPhone Mirroring window on a Mac, tracks the opponent's cards and elixir, and shows one short call at a time next to the phone: what to play, where, and why.

## Goals

- Help a casual player win more and pick up why the moves work.
- One glanceable call at a time, with a reason of five words or fewer. Example: `Bandit, left bridge: they're at 2 elixir`.
- Know the opponent's exact hand once all 8 of their cards have shown up.
- Pick a deck from the player's own cards and levels that gets the most out of the tracking.

## Non-goals

- Playing for the user. The helper never clicks or touches the phone.
- 2v2, draft, events and other modes. 1v1 trophy matches only.
- Drawing placement on the arena. Placement is given in words.
- Post-match review and trained decision models.

## How it works

```
iPhone Mirroring window
  -> capture.py   grab the window at full resolution, even when covered
  -> screen.py    crop to one canonical 664x1440 screen so every region is fixed
  -> vision.py    timer, double elixir, own hand and elixir, enemy deploys
  -> tracker.py   opponent card cycle and elixir
  -> advisor.py   one call: action, card, place, reason
  -> panel.py     window to the left of the phone
```

### Card cycle

Decks hold 8 cards: 4 in hand, 4 in a queue. A played card goes to the back of the queue and the front card enters the hand. Only the opening order is random, so the opponent's hand is their seen cards minus the last 4 plays, and the next card they draw is the one played 4 plays ago. Champions leave the cycle while alive and rejoin at the back when they die.

### Elixir

Both players start at 5 with a cap of 10 and gain 1 elixir every 2.8 seconds. The rate doubles when the double-elixir badge is on screen. The opponent's bar is simulated from the match clock and drained by each play. A play costing more than the estimate proves the estimate was low, so it is raised to that cost before draining, which keeps a missed detection from compounding.

### Seeing enemy plays

The game prints a card's name only for the player's own drag preview. Enemy units are recognized from the arena:

- Enemy units carry crimson level badges, the player's units carry blue ones. A new crimson badge cluster marks an enemy deploy, its size gives the unit count.
- The unit type comes from the open-source Clash Royale unit detector in [ClashRoyaleBuildABot](https://github.com/Pbatch/ClashRoyaleBuildABot), voted over the frames after the deploy and restricted to the opponent's known deck once it is complete.
- Spells leave no badges and are not in the detector's classes, so they are recognized separately.

The detector weights are AGPL-3.0, so they are downloaded on first run rather than shipped in this repo.

### Advice rules

Calls come from card facts (cost, what it targets, splash, swarm, air, building, spell size), not a trained model. In priority order:

1. **Defend** an enemy push on your side with the best affordable counter.
2. **Punish** when the opponent spent big and is low: attack the other lane.
3. **Safe window** when their answer to your win condition is out of hand.
4. **Hold** a card when their counter to it is in hand.
5. **Don't leak** at 10 elixir: play something cheap at the back.
6. Otherwise **wait** and build elixir.

Until all 8 opponent cards are known, calls that depend on their hand fall back to safer general advice.

### Deck pick

Reads the player's cards and levels from the official Clash Royale API and scores a short list of proven decks by how well the player's levels cover them and how much each benefits from tracking.

## Panel

A small always-on-top window parked to the left of the phone, which itself is moved against the right edge of the screen. Top: the current call in large type. Below: the opponent's hand and next card, and both elixir counts.

## Testing

Recorded matches are replayed through the same pipeline as live frames. Pure logic (cycle, elixir, advice) has unit tests. Recordings stay out of git.
