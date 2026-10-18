"""Runs inside the KataCR environment: arena images in on stdin, tracked units out on stdout.

Each request is a 4 byte big endian length followed by a JPEG of the arena crop.
Each reply is one JSON line, a list of [name, side, conf, x1, y1, x2, y2, track]
in arena crop pixels, side 1 is the opponent and track is -1 when untracked.
"""

from __future__ import annotations

import json
import os
import struct
import sys

import cv2
import numpy as np

ROOT = os.environ["KATACR_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "KataCR"))

from katacr.constants.label_list import idx2unit  # noqa: E402
from katacr.yolov8.combo_detect import ComboDetector  # noqa: E402


def main() -> None:
    # libraries print banners and warnings, the pipe carries replies only
    sys.stdout = sys.stderr
    # KATACR_WEIGHTS swaps in another pair of detectors, research/colab_train.py makes fine tuned ones
    weights = [os.path.join(ROOT, "KataCR", "runs", f"detector{i}_v0.7.13.pt") for i in (1, 2)]
    if os.environ.get("KATACR_WEIGHTS"):
        weights = os.environ["KATACR_WEIGHTS"].split(",")
    detector = ComboDetector(weights, tracker="bytetrack")
    for model in detector.models:
        model.overrides["device"] = os.environ.get("KATACR_DEVICE", "mps")
    stdout = os.fdopen(os.dup(1), "w")
    stdin = sys.stdin.buffer
    stdout.write("ready\n")
    stdout.flush()
    while True:
        header = stdin.read(4)
        if len(header) < 4:
            return
        (size,) = struct.unpack(">I", header)
        image = cv2.imdecode(np.frombuffer(stdin.read(size), np.uint8), cv2.IMREAD_COLOR)
        data = detector.infer(image).get_data()
        data = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        rows = []
        for row in data:
            track = int(row[4]) if len(row) == 8 else -1
            conf, cls, side = float(row[-3]), int(row[-2]), int(row[-1])
            rows.append([idx2unit[cls], side, round(conf, 3), *(round(float(v), 1) for v in row[:4]), track])
        stdout.write(json.dumps(rows) + "\n")
        stdout.flush()


if __name__ == "__main__":
    main()
