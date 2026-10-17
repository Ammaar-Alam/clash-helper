"""Render docs/demo.gif: a recorded match beside the panel the helper showed for it.

    python scripts/demo.py captures/match1.mp4 --start 80 --seconds 30
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from clash_helper.cli import video_frames
from clash_helper.game import Match
from clash_helper.panel import HEIGHT as HEIGHT_PT
from clash_helper.panel import WIDTH, Panel

OUT = Path(__file__).resolve().parents[1] / "docs/demo.gif"


def panel_image() -> np.ndarray:
    """Grab the panel window, it is the 320 pt wide window this process owns."""
    import Quartz

    for w in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID):
        if w.get("kCGWindowOwnerPID") == os.getpid() and int(w["kCGWindowBounds"]["Width"]) == WIDTH:
            image = Quartz.CGWindowListCreateImage(
                Quartz.CGRectNull,
                Quartz.kCGWindowListOptionIncludingWindow,
                w["kCGWindowNumber"],
                Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageBestResolution,
            )
            if image is None:
                continue
            width, height = Quartz.CGImageGetWidth(image), Quartz.CGImageGetHeight(image)
            stride = Quartz.CGImageGetBytesPerRow(image)
            data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image))
            return np.frombuffer(data, np.uint8).reshape(height, stride // 4, 4)[:, :width, :3].copy()
    raise RuntimeError("panel window not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--speed", type=float, default=2.0)
    args = parser.parse_args()

    panel = Panel(40, 60, HEIGHT_PT, queue.Queue())
    match = Match()

    def render() -> None:
        time.sleep(2.5)  # web fonts
        with tempfile.TemporaryDirectory() as tmp:
            index = 0
            for frame, now in video_frames(args.video, 5):
                view = match.step(frame, now)
                if now < args.start:
                    continue
                if now > args.start + args.seconds:
                    break
                panel.show(view)
                time.sleep(0.12)
                side = panel_image()
                height = side.shape[0]
                phone = cv2.resize(frame, (round(height * frame.shape[1] / frame.shape[0]), height))
                cv2.imwrite(f"{tmp}/{index:05d}.png", np.hstack([side, phone]))
                index += 1
            fps = 5 * args.speed
            palette = f"{tmp}/palette.png"
            frames = ["-framerate", str(fps), "-i", f"{tmp}/%05d.png"]
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *frames,
                            "-vf", "scale=560:-1:flags=lanczos,palettegen=stats_mode=diff", palette], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *frames, "-i", palette,
                            "-lavfi", "scale=560:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=sierra2_4a", str(OUT)],
                           check=True)
        panel.window.destroy()

    panel.run(render)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
