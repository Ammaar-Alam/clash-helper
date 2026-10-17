"""Official Clash Royale API, reached through the RoyaleAPI relay so the key survives IP changes."""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://proxy.royaleapi.dev/v1"
ROOT = Path(__file__).resolve().parents[2]


def setting(name: str) -> str | None:
    """A value from the environment or the repo .env file."""
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip()
    return None


def token() -> str | None:
    return setting("CR_API_TOKEN")


class ApiError(RuntimeError):
    pass


def get(path: str, **params) -> dict:
    key = token()
    if not key:
        raise ApiError("CR_API_TOKEN is missing, add it to .env")
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            # the relay's own hiccups are 5xx, worth another try
            if error.code >= 500 and attempt < 2:
                time.sleep(1 + attempt)
                continue
            detail = error.read().decode(errors="replace")[:200]
            raise ApiError(f"{error.code} on {path}: {detail}") from error
        except (urllib.error.URLError, http.client.IncompleteRead, TimeoutError) as error:
            # the relay sometimes cuts long replies short
            if attempt == 2:
                raise ApiError(f"network error on {path}: {error}") from error
    raise AssertionError("unreachable")


def tag_path(tag: str) -> str:
    return urllib.parse.quote("#" + tag.strip().lstrip("#").upper())


def player(tag: str) -> dict:
    return get(f"/players/{tag_path(tag)}")


def battlelog(tag: str) -> list[dict]:
    return get(f"/players/{tag_path(tag)}/battlelog")  # type: ignore[return-value]


def cards() -> list[dict]:
    return get("/cards")["items"]


def find_player(name: str, clan: str) -> dict | None:
    """Look up an opponent from the name and clan shown on screen, None if not unique."""
    clans = [c for c in get("/clans", name=clan, limit=20).get("items", []) if c["name"].lower() == clan.lower()]
    matches = []
    for c in clans:
        members = get(f"/clans/{tag_path(c['tag'])}/members").get("items", [])
        matches += [m for m in members if m["name"].lower() == name.lower()]
    if len(matches) != 1:
        return None
    return player(matches[0]["tag"])
