"""A see-through layer over the phone that shows where to drop the next card.

It ignores the mouse so dragging works straight through it, and it is kept
just above the iPhone Mirroring window instead of above every app.
"""

from __future__ import annotations

import cv2
import numpy as np

from clash_helper import screen
from clash_helper.capture import Window
from clash_helper.panel import card_image

RING = 46  # canonical px, about a tile and a half
CALM = (255, 236, 244, 235)  # BGRA, a pale violet white
URGENT = (90, 140, 255, 245)  # BGRA, the panel's warm red


def render(width: int, height: int, rect: tuple[int, int, int, int], card: str | None,
           target: tuple[int, int] | None, urgent: bool) -> np.ndarray:
    """Transparent BGRA image the size of the window capture with the ghost card drawn on."""
    image = np.zeros((height, width, 4), np.uint8)
    if target is None:
        return image
    rx, ry, rw, rh = rect
    scale = rw / screen.WIDTH
    x, y = round(rx + target[0] * scale), round(ry + target[1] * rh / screen.HEIGHT)
    color = URGENT if urgent else CALM
    radius = round(RING * scale)
    cv2.circle(image, (x, y), radius, color, max(2, round(3 * scale)), cv2.LINE_AA)
    cv2.circle(image, (x, y), max(3, round(5 * scale)), color, -1, cv2.LINE_AA)
    art = card_image(card) if card else None
    if art is not None:
        w = round(64 * scale)
        h = round(w * art.shape[0] / art.shape[1])
        ghost = cv2.resize(art, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        ghost[:, :, 3] *= 0.8
        # the ghost floats just above the spot so the ring stays visible
        x0, y0 = x - w // 2, y - radius - h - round(4 * scale)
        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(width, x0 + w), min(height, y0 + h)
        if x2 > x1 and y2 > y1:
            image[y1:y2, x1:x2] = ghost[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0].astype(np.uint8)
    return image


class Overlay:
    """Must be created and driven on the main thread, use AppHelper.callAfter from others."""

    def __init__(self) -> None:
        from AppKit import NSBackingStoreBuffered, NSColor, NSImageView, NSWindow, NSWindowStyleMaskBorderless

        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (10, 10)), NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(NSColor.clearColor())
        self.window.setHasShadow_(False)
        self.window.setIgnoresMouseEvents_(True)
        self.view = NSImageView.alloc().initWithFrame_(((0, 0), (10, 10)))
        self.window.setContentView_(self.view)
        self.last = None

    def show(self, phone: Window, frame_size: tuple[int, int], rect: tuple[int, int, int, int],
             card: str | None, target: tuple[int, int] | None, urgent: bool) -> None:
        from AppKit import NSImage, NSScreen, NSWindowAbove
        from Foundation import NSData

        state = (phone, frame_size, rect, card, target, urgent)
        if state != self.last:
            self.last = state
            width, height = frame_size
            bgra = render(width, height, rect, card, target, urgent)
            ok, png = cv2.imencode(".png", bgra)
            image = NSImage.alloc().initWithData_(NSData.dataWithBytes_length_(png.tobytes(), len(png)))
            image.setSize_((phone.width, phone.height))
            self.view.setImage_(image)
            # Cocoa counts y from the bottom of the main screen
            top = NSScreen.screens()[0].frame().size.height
            self.window.setFrame_display_(((phone.x, top - phone.y - phone.height), (phone.width, phone.height)), True)
            self.view.setFrame_(((0, 0), (phone.width, phone.height)))
        # sit right above the phone, so other apps still cover both
        self.window.orderWindow_relativeTo_(NSWindowAbove, phone.id)
