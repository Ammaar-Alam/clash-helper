"""Contact sheets of a recording for looking at plays by eye.

    python scripts/frames.py VIDEO START END [--step 0.4] [--box x0 y0 x1 y1] [--scale 1.0] [--out sheet.png]

Times are seconds into the video. The box is in canonical screen pixels (664x1440), the
default is the whole arena. Each tile is labelled with its time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from clash_helper.cli import video_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("start", type=float)
    parser.add_argument("end", type=float)
    parser.add_argument("--step", type=float, default=0.4)
    parser.add_argument("--box", type=int, nargs=4, default=[0, 140, 664, 1180])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("sheet.png"))
    args = parser.parse_args()
    x0, y0, x1, y1 = args.box
    tiles, next_time = [], args.start
    for frame, now in video_frames(args.video, max(5.0, 1 / args.step)):
        if now < next_time - 0.02:
            continue
        if now > args.end:
            break
        next_time = now + args.step
        tile = frame[y0:y1, x0:x1]
        tile = cv2.resize(tile, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA if args.scale < 1 else cv2.INTER_CUBIC)
        cv2.rectangle(tile, (0, 0), (92, 22), (0, 0, 0), -1)
        cv2.putText(tile, f"{now:.1f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        raise SystemExit("no frames in that range")
    while len(tiles) % args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + args.cols]) for i in range(0, len(tiles), args.cols)]
    cv2.imwrite(str(args.out), np.vstack(rows))
    print(f"wrote {args.out} with {len([t for t in tiles if t.any()])} frames")


if __name__ == "__main__":
    main()
