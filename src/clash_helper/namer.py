"""Ask a fast vision model which card the opponent just deployed."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

import cv2
import numpy as np

from clash_helper import api

MODEL = "gemini-3.5-flash-lite"
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
CROP = 180  # px around the landing spot on the canonical screen
NONE = "none"

PROMPT = """These are crops from a Clash Royale match, 0.4 seconds apart, centered where the opponent just deployed a card (a red stopwatch marks the spot in the first frames).
Opponent units wear crimson level badges with a number. Units with blue badges belong to the player, ignore them, and ignore older units that were already there.
Which of these opponent cards was deployed? Options (card, badge level): {options}
Answer "none" if no new opponent unit appears."""


def strip(frames: list[np.ndarray], x: int, y: int) -> np.ndarray:
    """The landing spot over several frames side by side."""
    half = CROP // 2
    tiles = []
    for frame in frames:
        padded = cv2.copyMakeBorder(frame, CROP, CROP, CROP, CROP, cv2.BORDER_CONSTANT)
        # units stand on the timer and their badges float above, so look a bit higher
        cy, cx = y + CROP - 20, x + CROP
        tiles.append(padded[cy - half : cy + half, cx - half : cx + half])
    return np.hstack(tiles)


def name_card(image: np.ndarray, options: dict[str, int | None]) -> str | None:
    """One of `options` or None, options map card name to its badge level if known."""
    key = api.setting("GEMINI_API_KEY")
    if not key:
        return None
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    listed = ", ".join(f"{n} ({lvl})" if lvl else n for n, lvl in options.items())
    body = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg.tobytes()).decode()}},
                    {"text": PROMPT.format(options=listed)},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {"card": {"type": "STRING", "enum": [*options, NONE]}},
                "required": ["card"],
            },
        },
    }
    request = urllib.request.Request(
        URL.format(model=MODEL),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(request, timeout=6) as response:
            reply = json.load(response)
        answer = json.loads(reply["candidates"][0]["content"]["parts"][0]["text"])["card"]
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as error:
        print(f"card namer failed: {error}", flush=True)
        return None
    return None if answer == NONE else answer
