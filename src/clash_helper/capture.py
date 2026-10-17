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


def visible_screen() -> tuple[int, int, int, int]:
    """Main screen area below the menu bar as left, top, right, bottom in points."""
    from AppKit import NSScreen

    screen = NSScreen.mainScreen()
    full, vis = screen.frame(), screen.visibleFrame()
    # AppKit is bottom-left origin, flip to the top-left origin everything else uses
    top = int(full.size.height - vis.origin.y - vis.size.height)
    return int(vis.origin.x), top, int(vis.origin.x + vis.size.width), int(top + vis.size.height)


def place_window(margin: int = 12) -> Window | None:
    """Park the phone against the right edge so the panel fits on its left."""
    window = find_window()
    if window is None:
        return None
    _, top, right, _ = visible_screen()
    script = f"""
    tell application "System Events" to tell process "{OWNER}"
        set frontmost to true
        set position of window 1 to {{{right - window.width - margin}, {top + margin}}}
    end tell
    """
    subprocess.run(["osascript", "-e", script], capture_output=True, check=False)
    time.sleep(0.4)
    return find_window()


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
