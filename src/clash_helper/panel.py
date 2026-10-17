"""Always-on-top window parked to the left of the phone."""

from __future__ import annotations

import queue
import tkinter as tk

from clash_helper.cards import card
from clash_helper.game import View

WIDTH = 300
BG, CARD_BG, TEXT, MUTED, ACCENT, URGENT = "#14161c", "#20242d", "#f2f3f5", "#8b93a3", "#c86bff", "#ff5d73"
FONT = "Helvetica Neue"


class Panel:
    def __init__(self, x: int, y: int, height: int, views: queue.Queue[View]) -> None:
        self.views = views
        self.root = tk.Tk()
        self.root.title("Clash Helper")
        self.root.geometry(f"{WIDTH}x{height}+{max(0, x)}+{y}")
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.headings: list[tk.Label] = []

        pad = {"padx": 18, "anchor": "w"}
        self.action = tk.Label(self.root, font=(FONT, 22, "bold"), fg=TEXT, bg=BG, wraplength=WIDTH - 36, justify="left")
        self.action.pack(pady=(22, 2), **pad)
        self.reason = tk.Label(self.root, font=(FONT, 15), fg=MUTED, bg=BG, wraplength=WIDTH - 36, justify="left")
        self.reason.pack(pady=(0, 22), **pad)

        self._heading("Their hand")
        self.hand = tk.Frame(self.root, bg=BG)
        self.hand.pack(fill="x", padx=18)
        self.next = tk.Label(self.root, font=(FONT, 13), fg=MUTED, bg=BG)
        self.next.pack(pady=(8, 0), **pad)
        self.maybe = tk.Label(self.root, font=(FONT, 12), fg=MUTED, bg=BG, wraplength=WIDTH - 36, justify="left")
        self.maybe.pack(pady=(2, 18), **pad)

        self._heading("Elixir")
        self.elixir = tk.Label(self.root, font=(FONT, 17, "bold"), fg=TEXT, bg=BG)
        self.elixir.pack(pady=(2, 18), **pad)

        self._heading("Their last plays")
        self.plays = tk.Label(self.root, font=(FONT, 12), fg=MUTED, bg=BG, wraplength=WIDTH - 36, justify="left")
        self.plays.pack(**pad)

        self.root.after(100, self._poll)

    def _heading(self, text: str) -> None:
        label = tk.Label(self.root, text=text.upper(), font=(FONT, 11, "bold"), fg=BG, bg=BG)
        label.pack(padx=18, anchor="w")
        self.headings.append(label)

    def _poll(self) -> None:
        view = None
        while not self.views.empty():
            view = self.views.get_nowait()
        if view is not None:
            self.show(view)
        self.root.after(100, self._poll)

    def show(self, view: View) -> None:
        self.action.config(text=view.call.action, fg=URGENT if view.call.urgent else TEXT)
        self.reason.config(text=view.call.reason)
        for child in self.hand.winfo_children():
            child.destroy()
        for heading in self.headings:
            # headings blend into the background until there is something under them
            heading.config(fg=ACCENT if view.in_match else BG)
        if not view.in_match:
            self.next.config(text="")
            self.maybe.config(text="")
            self.elixir.config(text="")
            self.plays.config(text="")
            return
        for i, name in enumerate(view.their_hand):
            chip = tk.Label(
                self.hand,
                text=f"{name}  {card(name).cost or ''}".strip(),
                font=(FONT, 13, "bold"),
                fg=TEXT,
                bg=CARD_BG,
                padx=8,
                pady=6,
                anchor="w",
            )
            chip.grid(row=i // 2, column=i % 2, sticky="ew", padx=(0, 6), pady=3)
        self.hand.columnconfigure(0, weight=1)
        self.hand.columnconfigure(1, weight=1)
        if not view.their_hand:
            tk.Label(self.hand, text="Not known yet", font=(FONT, 13), fg=MUTED, bg=BG).grid(row=0, column=0, sticky="w")
        self.next.config(text=f"Next in: {view.next_card}" if view.next_card else "")
        self.maybe.config(text=f"Could have: {', '.join(view.maybe)}" if view.maybe else "")
        self.elixir.config(text=f"Them {view.their_elixir:.1f}    You {view.my_elixir}")
        self.plays.config(text="  >  ".join(view.plays))

    def run(self) -> None:
        self.root.mainloop()
