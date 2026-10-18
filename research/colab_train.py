"""Fine tune KataCR's two unit detectors on frames from labeled games, on a Colab GPU VM.

Runs on the VM, started by research/colab_detector.sh. Reads /content/data (one folder per recording
from research/detector_data.py, and jobs.json), writes /content/out/<job>_detector<1|2>.pt.

First the base detector labels every crop with its own boxes, and the boxes of units that belong to a
labeled play (fixes.json, matched by place) get the class the true card gives them and are marked theirs.
Then each job trains on some recordings and keeps the others to check on, so every recording can be
judged by a detector that never saw it. Finished steps are skipped when the script runs again.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DATA = Path("/content/data")
OUT = Path("/content/out")
KATACR = Path("/content/KataCR")
COMMIT = "36ceb9fcfbd117c2ce3d97eacee435c1898eb7b8"  # the KataCR the helper installs
EPOCHS = int(os.environ.get("EPOCHS", "12"))
FREEZE = 10  # the backbone stays as trained, the neck and head learn these games
CONF = 0.35  # the base detector's boxes kept as labels
ARENA = (14, 105, 637, 1008)  # clash_helper.kcr: the crop on the canonical screen, x, y, w, h
INPUT = (568, 896)
REACH = 10  # px on the canonical screen between a box and the scanned unit it is


def setup() -> None:
    if not (KATACR / "runs").exists():
        subprocess.run(["git", "clone", "https://github.com/wty-yy/KataCR.git", str(KATACR)], check=True)
        subprocess.run(["git", "-C", str(KATACR), "checkout", COMMIT], check=True)
        constant = KATACR / "katacr/build_dataset/constant.py"
        text = re.sub(r'^path_dataset = Path\(".*"\)$', 'path_dataset = Path("/content/dataset")', constant.read_text(),
                      count=1, flags=re.M)
        constant.write_text(text)
        # its loader fails on a frame without a box for that detector, its own data always had towers in view
        loader = KATACR / "katacr/yolov8/custom_dataset.py"
        old = "lb = np.concatenate((classes, xywh), 1)  # (cls, bel, xywh)"
        loader.write_text(loader.read_text().replace(old, old.replace("1)", "1) if len(lb) else np.zeros((0, 6), np.float32)")))
        (KATACR / "runs").mkdir(exist_ok=True)
        for name in ("detector1_v0.7.13.pt", "detector2_v0.7.13.pt"):
            shutil.copy(DATA / name, KATACR / "runs" / name)
    Path("/content/dataset").mkdir(exist_ok=True)


def label(recordings: list[str]) -> None:
    """The base detector's boxes for every crop, with the fixes for labeled plays applied."""
    import cv2
    import numpy as np
    from katacr.constants.label_list import idx2unit, unit2idx
    from katacr.yolov8.combo_detect import ComboDetector

    detector = None
    sx, sy = ARENA[2] / INPUT[0], ARENA[3] / INPUT[1]
    for r in recordings:
        folder = DATA / r
        if (folder / "labelled").exists():
            continue
        if detector is None:
            detector = ComboDetector([str(KATACR / f"runs/detector{i}_v0.7.13.pt") for i in (1, 2)], conf=CONF, tracker=None)
            for m in detector.models:
                m.overrides["device"] = 0
        fixes = json.loads((folder / "fixes.json").read_text())
        matched = wanted = 0
        for line in (folder / "list.txt").read_text().split():
            data = detector.infer(cv2.imread(str(DATA / line))).get_data()
            data = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
            boxes = [[idx2unit[int(row[-2])], int(row[-1]), *row[:4]] for row in data]
            for name, x, bottom, fixed, side in fixes.get(Path(line).stem, []):
                wanted += 1
                near = [(abs(ARENA[0] + (b[2] + b[4]) / 2 * sx - x) + abs(ARENA[1] + b[5] * sy - bottom), i)
                        for i, b in enumerate(boxes)]
                d, i = min(near, default=(REACH * 3, -1))
                if d <= REACH * 2:
                    boxes[i][0], boxes[i][1] = fixed or name, side
                    matched += 1
            rows = [f"{unit2idx[n]} {(x1 + x2) / 2 / INPUT[0]:.6f} {(y1 + y2) / 2 / INPUT[1]:.6f} "
                    f"{(x2 - x1) / INPUT[0]:.6f} {(y2 - y1) / INPUT[1]:.6f} {bel}" for n, bel, x1, y1, x2, y2 in boxes]
            # KataCR's loader takes a frame without a label file as background but fails on an empty one
            if rows:
                (DATA / line).with_suffix(".txt").write_text("\n".join(rows))
        print(f"labelled {r}: {matched} of {wanted} play units matched a box", flush=True)
        (folder / "labelled").touch()


def image_list(name: str, recordings: list[str], every: int = 1) -> str:
    lines = [f"./{line}" for r in recordings
             for i, line in enumerate((DATA / r / "list.txt").read_text().split()) if i % every == 0]
    (DATA / f"{name}.txt").write_text("\n".join(lines))
    return f"{name}.txt"


def main() -> None:
    setup()
    sys.path.insert(0, str(KATACR))
    os.chdir(KATACR)
    import yaml
    from katacr.yolov8.train import YOLO_CR
    from ultralytics.cfg import get_cfg

    jobs = json.loads((DATA / "jobs.json").read_text())
    OUT.mkdir(exist_ok=True)
    label(sorted({r for spec in jobs.values() for r in spec["train"] + spec["check"]}))
    for job, spec in jobs.items():
        for det in (1, 2):
            name = f"{job}_detector{det}"
            if (OUT / f"{name}.pt").exists():
                continue
            names = yaml.safe_load((KATACR / f"katacr/yolov8/detector{det}/data.yaml").read_text())["names"]
            data = {"path": str(DATA), "train": image_list(f"{job}_train", spec["train"]),
                    "val": image_list(f"{job}_val", spec["check"], every=4), "names": names}
            (DATA / f"{name}.yaml").write_text(yaml.safe_dump(data))
            weights = str(KATACR / f"runs/detector{det}_v0.7.13.pt")
            cfg = dict(get_cfg("./katacr/yolov8/ClashRoyale.yaml"))
            cfg.update(data=str(DATA / f"{name}.yaml"), model=weights, pretrained=True, epochs=EPOCHS, patience=EPOCHS,
                       batch=16, device=0, workers=8, freeze=FREEZE, optimizer="AdamW", lr0=1e-4, deterministic=False,
                       plots=False, project=str(OUT / "runs"), name=name, exist_ok=True)
            print(f"training {name} on {len(spec['train'])} recordings", flush=True)
            YOLO_CR(weights).train(**cfg)
            # the last epoch, the best one would be chosen on recordings meant for judging
            shutil.copy(OUT / "runs" / name / "weights" / "last.pt", OUT / f"{name}.tmp")
            os.replace(OUT / f"{name}.tmp", OUT / f"{name}.pt")
            print(f"done {name}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
