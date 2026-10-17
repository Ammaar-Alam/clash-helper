from __future__ import annotations

import argparse
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from clash_helper import capture, screen


def record(out_dir: Path, fps: float) -> None:
    """Save frames until Ctrl+C, used to build test fixtures from real matches."""
    window = capture.find_window()
    if window is None:
        raise SystemExit("iPhone Mirroring is not open")
    folder = out_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder.mkdir(parents=True, exist_ok=True)
    print(f"recording to {folder}, Ctrl+C to stop", flush=True)
    start = time.monotonic()
    count = 0
    try:
        while True:
            tick = time.monotonic()
            frame = capture.grab(window)
            if frame is None:
                window = capture.find_window() or window
            else:
                # elapsed ms in the name keeps real timing for elixir checks
                ms = int((tick - start) * 1000)
                cv2.imwrite(str(folder / f"{ms:07d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                count += 1
            time.sleep(max(0.0, 1 / fps - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print(f"\nsaved {count} frames", flush=True)


def video_frames(path: Path, fps: float):
    """Frames of a screen recording at `fps`, with their time in seconds."""
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise SystemExit(f"cannot open {path}")
    step = max(1, round(video.get(cv2.CAP_PROP_FPS) / fps))
    index = 0
    while True:
        ok, frame = video.read()
        if not ok:
            return
        if index % step == 0:
            canonical = screen.normalize(frame)
            if canonical is not None:
                yield canonical, index / video.get(cv2.CAP_PROP_FPS)
        index += 1


def replay(path: Path, fps: float) -> None:
    """Print every opponent play and call change from a recorded match."""
    from clash_helper.game import Match

    match = Match(sync=True)
    last_call = ""
    plays = 0
    for frame, now in video_frames(path, fps):
        view = match.step(frame, now)
        if match.played > plays:
            plays = match.played
            print(f"{now:6.1f}s  they played {match.last_play:<16} elixir {view.their_elixir:4.1f}  hand {view.their_hand}")
        if str(view.call) != last_call:
            last_call = str(view.call)
            print(f"{now:6.1f}s    -> {last_call}")


def live_frames(fps: float):
    """Frames from the phone, with where the screen sits in the window for the overlay."""
    window = capture.find_window()
    start = time.monotonic()
    while True:
        tick = time.monotonic()
        frame = capture.grab(window) if window else None
        rect = screen.screen_rect(frame) if frame is not None else None
        if rect is None:
            # the window was closed, moved to another space or resized
            window = capture.find_window()
        else:
            yield screen.normalize(frame, rect), tick - start, (window, (frame.shape[1], frame.shape[0]), rect)
        time.sleep(max(0.0, 1 / fps - (time.monotonic() - tick)))


def _on_main(make):
    """Build an AppKit object on the main thread and hand it back."""
    from PyObjCTools import AppHelper

    made: list = []
    done = threading.Event()
    AppHelper.callAfter(lambda: (made.append(make()), done.set()))
    done.wait()
    return made[0]


def run(video: Path | None, fps: float) -> None:
    from PyObjCTools import AppHelper

    from clash_helper.game import Match
    from clash_helper.journal import Journal
    from clash_helper.overlay import Overlay
    from clash_helper.panel import WIDTH, Panel

    window = capture.place_window() or capture.find_window()
    if window is None:
        raise SystemExit("Open iPhone Mirroring first")
    panel = Panel(window.x - WIDTH - 12, window.y, window.height, queue.Queue())

    def loop() -> None:
        match = Match()
        overlay = _on_main(Overlay) if not video else None
        journal = Journal(fps) if not video else None
        start = time.monotonic()
        frames = ((f, t, None) for f, t in video_frames(video, fps)) if video else live_frames(fps)
        for frame, now, geometry in frames:
            if video:
                time.sleep(max(0.0, now - (time.monotonic() - start)))
            view = match.step(frame, now)
            panel.show(view)
            if journal:
                journal.update(match, frame, view, now)
            if overlay and geometry:
                phone, size, rect = geometry
                target = view.call.target if view.in_match else None
                AppHelper.callAfter(overlay.show, phone, size, rect, view.call.card, target, view.call.urgent)

    panel.run(loop)


def main() -> None:
    parser = argparse.ArgumentParser(prog="clash-helper")
    parser.add_argument("--fps", type=float, default=6.0)
    sub = parser.add_subparsers(dest="command")
    rec = sub.add_parser("record", help="save frames from the mirrored phone")
    rec.add_argument("--out", type=Path, default=Path("captures"))
    rep = sub.add_parser("replay", help="run a screen recording through the helper")
    rep.add_argument("video", type=Path)
    rep.add_argument("--panel", action="store_true", help="show the panel at real speed instead of printing")
    sub.add_parser("deck", help="suggest a deck from your cards")
    sub.add_parser("setup", help="install the unit detector, once")
    args = parser.parse_args()
    if args.command == "record":
        record(args.out, args.fps)
    elif args.command == "setup":
        from clash_helper import kcr

        kcr.setup()
    elif args.command == "deck":
        from clash_helper import api, deck

        tag = api.setting("CR_PLAYER_TAG")
        if not tag or not api.token():
            raise SystemExit("set CR_API_TOKEN and CR_PLAYER_TAG in .env first")
        try:
            deck.suggest(tag)
        except api.ApiError as error:
            raise SystemExit(f"Clash Royale API: {error}") from None
    elif args.command == "replay" and not args.panel:
        replay(args.video, args.fps)
    else:
        run(getattr(args, "video", None), args.fps)


if __name__ == "__main__":
    main()
