"""Find, place and grab the iPhone Mirroring window."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

import numpy as np

OWNER = "iPhone Mirroring"


@dataclass(frozen=True, slots=True)
class Window:
    id: int
    x: int
    y: int
    width: int
    height: int


def find_window() -> Window | None:
    import Quartz

    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    ) or []
    # menus and popovers share the owner, the phone itself is the biggest layer 0 window
    phones = [w for w in windows if w.get("kCGWindowOwnerName") == OWNER and w.get("kCGWindowLayer") == 0]
    if not phones:
        return None
    w = max(phones, key=lambda w: w["kCGWindowBounds"]["Width"] * w["kCGWindowBounds"]["Height"])
    b = w["kCGWindowBounds"]
    return Window(int(w["kCGWindowNumber"]), int(b["X"]), int(b["Y"]), int(b["Width"]), int(b["Height"]))


def place_window() -> Window | None:
    """Window > Move & Resize > Right, the big phone view with room for the panel on its left."""
    if find_window() is None:
        return None
    script = f"""
    tell application "System Events" to tell process "{OWNER}"
        set frontmost to true
        delay 0.3
        click menu item "Right" of menu 1 of menu item "Move & Resize" of menu 1 of menu bar item "Window" of menu bar 1
    end tell
    """
    subprocess.run(["osascript", "-e", script], capture_output=True, check=False)
    # the window drops out of the list while it animates to the new size
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        time.sleep(0.3)
        window = find_window()
        if window is not None:
            return window
    return None


def grab(window: Window) -> np.ndarray | None:
    """BGR frame of the window at full pixel resolution, works even when covered."""
    import Quartz

    image = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull,
        Quartz.kCGWindowListOptionIncludingWindow,
        window.id,
        Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageBestResolution,
    )
    if image is None:
        return None
    width, height = Quartz.CGImageGetWidth(image), Quartz.CGImageGetHeight(image)
    stride = Quartz.CGImageGetBytesPerRow(image)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image))
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, stride // 4, 4)
    return np.ascontiguousarray(pixels[:, :width, :3])
