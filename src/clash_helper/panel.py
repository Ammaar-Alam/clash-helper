"""Always-on-top window parked to the left of the phone, drawn as a small web page."""

from __future__ import annotations

import base64
import json
import queue
from functools import cache
from pathlib import Path

import cv2

from clash_helper.cards import DATA, card
from clash_helper.game import View
from clash_helper.hand import icon_path

WIDTH = 320
HEIGHT = 700  # content height, the panel lines up with the top of the phone
HTML = Path(__file__).with_name("panel.html")


@cache
def _icon_urls() -> dict[str, str]:
    return {c["name"]: c["icon"] for c in json.loads(DATA.read_text()) if c.get("icon")}


@cache
def art(name: str) -> str | None:
    """Card art as a small data URI, keeping its transparent corners."""
    url = _icon_urls().get(name)
    if not url:
        return None
    try:
        rgba = cv2.imread(str(icon_path(url)), cv2.IMREAD_UNCHANGED)
    except OSError:
        return None
    if rgba is None:
        return None
    if rgba.shape[2] == 4:
        ys, xs = (rgba[:, :, 3] > 16).nonzero()
        rgba = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    rgba = cv2.resize(rgba, (128, round(128 * rgba.shape[0] / rgba.shape[1])), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", rgba)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode() if ok else None


class Panel:
    def __init__(self, x: int, y: int, height: int, views: queue.Queue[View]) -> None:
        import webview

        self.views = views
        self.sent: set[str] = set()
        self.window = webview.create_window(
            "Clash Helper",
            html=HTML.read_text(),
            x=max(0, x),
            y=y,
            width=WIDTH,
            height=min(height, HEIGHT),
            on_top=True,
            frameless=True,
            easy_drag=True,
            background_color="#17151d",
        )

    def payload(self, view: View) -> dict:
        waiting = [None if n in (None, "?") else n for n in view.queue]
        names = set(view.their_hand + view.maybe + [n for n in waiting if n])
        fresh = {n: uri for n in names - self.sent if (uri := art(n))}
        self.sent |= fresh.keys()
        return {
            "inMatch": view.in_match,
            "opponent": view.opponent,
            "action": view.call.action,
            "reason": view.call.reason,
            "urgent": view.call.urgent,
            "hand": view.their_hand,
            "queue": waiting,
            "maybe": view.maybe,
            "costs": {n: card(n).cost for n in names},
            "them": round(view.their_elixir, 2),
            "you": view.my_elixir,
            "deckKnown": view.deck_known,
            "art": fresh,
        }

    def show(self, view: View) -> None:
        self.window.evaluate_js(f"render({json.dumps(self.payload(view))})")

    def _pump(self) -> None:
        while True:
            view = self.views.get()
            # only the newest state matters, skip any backlog
            while not self.views.empty():
                view = self.views.get_nowait()
            self.show(view)

    def run(self, loop=None) -> None:
        """Start the window, `loop` replaces the default queue pump when given."""
        import webview

        webview.start(loop or self._pump)
