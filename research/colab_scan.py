"""Scan recordings with a pair of detectors on the Colab GPU, the same rows as scripts/scan.py.

    python colab_scan.py OUT_DIR WEIGHTS1,WEIGHTS2 VIDEO...

Runs on the VM next to research/colab_train.py, with clash_helper's source in /content/src. Each frame
goes through the same crop, JPEG round trip and tracker as the helper's detector worker, so a scan made
here differs from one made on the Mac only by the weights and the GPU.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/content/src")
sys.path.insert(0, "/content/KataCR")
from clash_helper import kcr, screen  # noqa: E402
from katacr.constants.label_list import idx2unit  # noqa: E402
from katacr.yolov8.combo_detect import ComboDetector  # noqa: E402

FPS = 6.0


def frames(path: Path):
    """clash_helper.cli.video_frames, without the Mac capture code that module imports."""
    video = cv2.VideoCapture(str(path))
    rate = video.get(cv2.CAP_PROP_FPS)
    step, index = max(1, round(rate / FPS)), 0
    while True:
        ok, frame = video.read()
        if not ok:
            return
        if index % step == 0:
            canonical = screen.normalize(frame)
            if canonical is not None:
                yield canonical, index / rate
        index += 1


def scan(video: Path, out: Path, weights: list[str]) -> None:
    detector = ComboDetector(weights, tracker="bytetrack")
    for m in detector.models:
        m.overrides["device"] = 0
    x0, y0, w, h = kcr.ARENA
    sx, sy = w / kcr.INPUT[0], h / kcr.INPUT[1]
    rows, started = [], time.time()
    for frame, now in frames(video):
        crop = cv2.resize(frame[y0 : y0 + h, x0 : x0 + w], kcr.INPUT, interpolation=cv2.INTER_AREA)
        image = cv2.imdecode(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1], cv2.IMREAD_COLOR)
        data = detector.infer(image).get_data()
        data = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        units = []
        for row in data:
            track = int(row[4]) if len(row) == 8 else -1
            conf, cls, side = round(float(row[-3]), 3), int(row[-2]), int(row[-1])
            x1, y1, x2, y2 = (round(float(v), 1) for v in row[:4])
            name = idx2unit[cls]
            units.append([name, kcr.card_of(name), int(side == 1), conf, round(x0 + (x1 + x2) / 2 * sx),
                          round(y0 + y1 * sy), round(y0 + y2 * sy), track])
        rows.append([round(now, 3), units])
    with gzip.open(out, "wt") as f:
        json.dump(rows, f)
    print(f"{out.name}: {len(rows)} frames in {time.time() - started:.0f}s", flush=True)


def main() -> None:
    out, weights = Path(sys.argv[1]), sys.argv[2].split(",")
    out.mkdir(parents=True, exist_ok=True)
    for video in sys.argv[3:]:
        scan(Path(video), out / f"{Path(video).stem}.json.gz", weights)


if __name__ == "__main__":
    main()
