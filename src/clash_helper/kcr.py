"""The KataCR unit detector, run as a background worker in its own environment.

`clash-helper setup` installs it. The worker sees every unit on the arena with
its side and a track id that stays with the unit across frames.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import threading
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np

from clash_helper.cards import all_cards

ROOT = Path.home() / "Library/Caches/clash-helper/katacr"
PYTHON = ROOT / ".venv/bin/python"
# the arena part of a 2.16 to 2.17 tall screen and the size the detectors were trained on
ARENA = (14, 105, 637, 1008)  # x, y, w, h on the canonical screen
INPUT = (568, 896)
# things the detector knows that are not cards
NOT_CARDS = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower", "dagger-duchess-tower-bar",
             "tower-bar", "king-tower-bar", "bar", "bar-level", "clock", "emote", "text", "elixir", "selected",
             "skeleton-king-bar", "evolution-symbol"}
ALIASES = {
    "minipekka": "Mini P.E.K.K.A", "pekka": "P.E.K.K.A", "rascalboy": "Rascals", "rascalgirl": "Rascals",
    "skeleton": "Skeletons", "goblin": "Goblins", "barbarian": "Barbarians", "elitebarbarian": "Elite Barbarians",
    "speargoblin": "Spear Goblins", "bat": "Bats", "minion": "Minions", "guard": "Guards", "wallbreaker": "Wall Breakers",
    "royalhog": "Royal Hogs", "royalrecruit": "Royal Recruits", "thelog": "The Log", "xbow": "X-Bow",
    "golemite": "Golem", "lavapup": "Lava Hound", "skeletondragon": "Skeleton Dragons", "zappy": "Zappies",
    "elixirgolembig": "Elixir Golem", "elixirgolemmid": "Elixir Golem", "elixirgolemsmall": "Elixir Golem",
    "phoenixbig": "Phoenix", "phoenixsmall": "Phoenix", "phoenixegg": "Phoenix", "royalguardian": "Little Prince",
    "axe": "Executioner", "bowl": "Bowler",
}


REPO = "https://github.com/wty-yy/KataCR.git"
COMMIT = "master"
WEIGHTS = {"detector1_v0.7.13.pt": "1DMD-EYXa1qn8lN4JjPQ7UIuOMwaqS5w_", "detector2_v0.7.13.pt": "1yEq-6liLhs_pUfipJM1E-tMj6l4FSbxD"}
# pinned to what the detectors were trained and tested with
PACKAGES = ["torch==2.2.2", "torchvision==0.17.2", "ultralytics==8.1.24", "numpy==1.26.4", "opencv-python==4.9.0.80",
            "lap", "Pillow", "scipy", "jax==0.4.26", "jaxlib==0.4.26", "flax==0.8.1", "optax==0.2.2",
            "orbax-checkpoint==0.5.3", "einops", "tensorboardX", "tqdm", "pandas", "matplotlib", "gdown"]


def ready() -> bool:
    return PYTHON.exists() and all((ROOT / "KataCR/runs" / name).exists() for name in WEIGHTS)


def setup() -> None:
    """Install KataCR, its weights and a Python environment for it, safe to run again."""
    def run(*cmd: str, cwd: Path | None = None) -> None:
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=cwd, check=True)

    ROOT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "KataCR").exists():
        run("git", "clone", "--depth", "1", REPO, str(ROOT / "KataCR"))
    # its constants insist on a dataset folder, point them at one next to it
    constant = ROOT / "KataCR/katacr/build_dataset/constant.py"
    text = constant.read_text()
    marker = 'path_dataset = Path(__file__).parents[3] / "Clash-Royale-Detection-Dataset"'
    if marker not in text:
        text = re.sub(r'^path_dataset = Path\(".*"\)$', marker, text, count=1, flags=re.M)
        constant.write_text(text)
    (ROOT / "Clash-Royale-Detection-Dataset").mkdir(exist_ok=True)
    if not PYTHON.exists():
        run("uv", "venv", "--python", "3.11", str(ROOT / ".venv"))
    run("uv", "pip", "install", "--python", str(PYTHON), *PACKAGES)
    runs = ROOT / "KataCR/runs"
    runs.mkdir(exist_ok=True)
    for name, file_id in WEIGHTS.items():
        if not (runs / name).exists():
            run(str(ROOT / ".venv/bin/gdown"), file_id, "-O", str(runs / name))
    print("unit detector ready" if ready() else "setup did not finish, see the errors above", flush=True)


@cache
def card_of(unit: str) -> str | None:
    """The card a detected unit or spell belongs to, None for towers, bars and other screen parts."""
    if unit in NOT_CARDS:
        return None
    base = re.sub(r"[^a-z]", "", unit.replace("-evolution", "").lower())
    names = {re.sub(r"[^a-z]", "", n.lower()): n for n in all_cards()}
    return ALIASES.get(base) or names.get(base) or names.get(base + "s")


@dataclass
class Unit:
    name: str  # detector class, like archer-evolution
    card: str | None
    enemy: bool
    conf: float
    x: int  # center, canonical screen pixels
    top: int
    bottom: int  # feet
    track: int


class Detector:
    """Send a frame when the worker is free, read the newest result whenever."""

    def __init__(self) -> None:
        env = {**os.environ, "KATACR_ROOT": str(ROOT), "PYTHONWARNINGS": "ignore"}
        self.proc = subprocess.Popen(
            [str(PYTHON), str(Path(__file__).with_name("kcr_worker.py"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
        )
        line = self.proc.stdout.readline()
        while line and line.strip() != b"ready":
            line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("the unit detector did not start, run clash-helper setup")
        self.busy = False
        self.sent_at = 0.0
        self.latest: tuple[float, list[Unit]] = (0.0, [])
        self.lock = threading.Lock()

    def submit(self, frame: np.ndarray, now: float) -> bool:
        """Queue a frame if the worker is idle, frames that arrive while it works are skipped."""
        if self.busy:
            return False
        x, y, w, h = ARENA
        crop = cv2.resize(frame[y : y + h, x : x + w], INPUT, interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return False
        self.busy, self.sent_at = True, now
        self.proc.stdin.write(struct.pack(">I", len(jpeg)) + jpeg.tobytes())
        self.proc.stdin.flush()
        threading.Thread(target=self._read, daemon=True).start()
        return True

    def _read(self) -> None:
        try:
            line = self.proc.stdout.readline()
            # anything that is not a reply is stray library output
            while line and not line.startswith(b"["):
                line = self.proc.stdout.readline()
            rows = json.loads(line) if line else []
            self._store(rows)
        finally:
            self.busy = False

    def _store(self, rows: list) -> None:
        x, y, w, h = ARENA
        sx, sy = w / INPUT[0], h / INPUT[1]
        units = [
            Unit(name, card_of(name), side == 1, conf, round(x + (x1 + x2) / 2 * sx), round(y + y1 * sy),
                 round(y + y2 * sy), track)
            for name, side, conf, x1, y1, x2, y2, track in rows
        ]
        with self.lock:
            self.latest = (self.sent_at, units)

    def result(self) -> tuple[float, list[Unit]]:
        """When the newest analysed frame was taken and what was on it."""
        with self.lock:
            return self.latest

    def close(self) -> None:
        self.proc.kill()
