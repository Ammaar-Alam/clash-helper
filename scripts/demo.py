"""Render docs/demo.gif: a recorded match beside the panel the helper showed for it.

    python scripts/demo.py captures/match1.mp4 --start 80 --seconds 30
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from clash_helper.cli import video_frames
from clash_helper.game import Match
from clash_helper.panel import Panel

OUT = Path(__file__).resolve().parents[1] / "docs/demo.gif"
HEIGHT = 720


def panel_image(panel: Panel) -> np.ndarray:
    """Draw the Tk panel into an image without screen capture permissions."""
    import Quartz

    panel.root.update()
    title = panel.root.title()
    for w in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID):
        if w.get("kCGWindowName") == title:
            image = Quartz.CGWindowListCreateImage(
                Quartz.CGRectNull,
                Quartz.kCGWindowListOptionIncludingWindow,
                w["kCGWindowNumber"],
                Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageBestResolution,
            )
            width, height = Quartz.CGImageGetWidth(image), Quartz.CGImageGetHeight(image)
            stride = Quartz.CGImageGetBytesPerRow(image)
            data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image))
            return np.frombuffer(data, np.uint8).reshape(height, stride // 4, 4)[:, :width, :3].copy()
    raise RuntimeError("panel window not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--speed", type=float, default=2.0)
    args = parser.parse_args()

    panel = Panel(40, 60, 760, queue.Queue())
    match = Match()
    with tempfile.TemporaryDirectory() as tmp:
        index = 0
        for frame, now in video_frames(args.video, 5):
            view = match.step(frame, now)
            if now < args.start:
                continue
            if now > args.start + args.seconds:
                break
            panel.show(view)
            side = panel_image(panel)
            phone = cv2.resize(frame, (round(HEIGHT * frame.shape[1] / frame.shape[0]), HEIGHT))
            side = cv2.resize(side, (round(HEIGHT * side.shape[1] / side.shape[0]), HEIGHT))
            cv2.imwrite(f"{tmp}/{index:05d}.png", np.hstack([side, phone]))
            index += 1
        panel.root.destroy()
        fps = 5 * args.speed
        palette = f"{tmp}/palette.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/%05d.png",
                        "-vf", "scale=720:-1:flags=lanczos,palettegen", palette], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/%05d.png",
                        "-i", palette, "-lavfi", "scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse", str(OUT)], check=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
