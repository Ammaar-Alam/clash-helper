"""Opponent card cycle and elixir model.

Clash Royale decks hold 8 cards, 4 in hand and 4 waiting in a queue. A played
card joins the back of the queue and the front of the queue enters the hand.
Only the opening order is random, so once all 8 cards have been seen the hand
is exactly the deck minus the last 4 plays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DECK_SIZE = 8
HAND_SIZE = 4
START_ELIXIR = 5.0
MAX_ELIXIR = 10.0
SECONDS_PER_ELIXIR = 2.8
UNKNOWN = "?"  # a play we saw land but could not name


@dataclass
class Opponent:
    plays: list[str] = field(default_factory=list)
    deck: list[str] = field(default_factory=list)
    # champions leave the cycle while alive and rejoin at the back on death
    benched: list[str] = field(default_factory=list)
    elixir: float = START_ELIXIR
    last_time: float = 0.0

    def tick(self, now: float, multiplier: float = 1.0) -> None:
        """Advance elixir to match time `now` in seconds at the current rate multiplier."""
        gained = (now - self.last_time) * multiplier / SECONDS_PER_ELIXIR
        self.elixir = min(MAX_ELIXIR, self.elixir + max(0.0, gained))
        self.last_time = now

    def play(self, card: str, cost: float, now: float, multiplier: float = 1.0, champion: bool = False) -> None:
        self.tick(now, multiplier)
        # a play we can see proves they had the elixir, so a missed play never leaves us low
        self.elixir = max(self.elixir, cost) - cost
        if card != UNKNOWN and card not in self.deck and len(self.deck) < DECK_SIZE:
            self.deck.append(card)
        if champion:
            self.benched.append(card)
        else:
            self.plays.append(card)

    def champion_died(self, card: str) -> None:
        if card in self.benched:
            self.benched.remove(card)
            self.plays.append(card)

    @property
    def queue(self) -> list[str | None]:
        """Waiting cards front first, None for opening cards not yet seen."""
        queue_size = HAND_SIZE - len(self.benched)
        recent = self.plays[-queue_size:] if queue_size > 0 else []
        return [None] * (queue_size - len(recent)) + recent

    @property
    def hand(self) -> list[str]:
        """Cards certainly in hand right now, all 4 once the deck and queue are known."""
        queue = self.queue
        waiting = set(queue) | set(self.benched)
        if UNKNOWN in queue:
            # ponytail: any hand card could be the unnamed one, so nothing is certain until it cycles out
            return []
        if None not in queue and self.deck_complete:
            return [card for card in self.deck if card not in waiting]
        # a card played earlier than the whole queue has cycled back into the hand
        return [card for card in self.deck if card in self.plays and card not in waiting]

    @property
    def maybe(self) -> list[str]:
        """Known deck cards that could be in hand or still waiting."""
        hand = self.hand
        waiting = set(self.queue) | set(self.benched)
        return [card for card in self.deck if card not in hand and card not in waiting]

    def preload(self, cards: list[str]) -> None:
        """Their full deck from the API, which overrules anything guessed so far."""
        self.deck = list(cards[:DECK_SIZE])
        self.plays = [p if p in self.deck else UNKNOWN for p in self.plays]
        self.benched = [b for b in self.benched if b in self.deck]

    @property
    def next_card(self) -> str | None:
        """Card that enters the hand after their next play, None if not seen yet."""
        queue = self.queue
        return queue[0] if queue else None

    @property
    def deck_complete(self) -> bool:
        return len(self.deck) == DECK_SIZE

    def plays_until_back(self, card: str) -> int | None:
        """0 if in hand, otherwise how many of their plays until it is back in hand."""
        if card in self.hand:
            return 0
        if card in self.queue:
            return self.queue.index(card) + 1
        return None
