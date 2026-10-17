"""Turn per-frame enemy badges into deploy events."""

from __future__ import annotations

from dataclasses import dataclass, field

MATCH_RADIUS = 45  # px a unit can move between frames at 5 fps
GROUP_RADIUS = 130  # units of one card land close together
GROUP_SECONDS = 0.8  # and within this long of each other
FORGET_SECONDS = 0.8  # badges flicker, keep a lost unit this long


@dataclass
class Unit:
    x: int
    y: int
    seen: float


@dataclass
class Deploy:
    time: float
    x: int
    y: int
    units: int = 1
    confirmed: bool = False


@dataclass
class DeployTracker:
    units: list[Unit] = field(default_factory=list)
    open: list[Deploy] = field(default_factory=list)

    def update(self, badges: list[tuple[int, int]], now: float) -> list[Deploy]:
        """Feed one frame, returns deploys that just became certain."""
        fresh: list[tuple[int, int]] = []
        for x, y in badges:
            near = [u for u in self.units if abs(u.x - x) + abs(u.y - y) < MATCH_RADIUS]
            if near:
                unit = min(near, key=lambda u: abs(u.x - x) + abs(u.y - y))
                unit.x, unit.y, unit.seen = x, y, now
            else:
                fresh.append((x, y))
                self.units.append(Unit(x, y, now))
        self.units = [u for u in self.units if now - u.seen <= FORGET_SECONDS]

        for x, y in fresh:
            group = next(
                (d for d in self.open if now - d.time <= GROUP_SECONDS and abs(d.x - x) + abs(d.y - y) < GROUP_RADIUS),
                None,
            )
            if group:
                group.units += 1
            else:
                self.open.append(Deploy(now, x, y))

        done = []
        for deploy in self.open:
            # a deploy is final once its grouping window closes, one frame blips are dropped
            if now - deploy.time > GROUP_SECONDS:
                alive = any(abs(u.x - deploy.x) + abs(u.y - deploy.y) < GROUP_RADIUS + 60 for u in self.units)
                if alive:
                    deploy.confirmed = True
                    done.append(deploy)
        self.open = [d for d in self.open if not d.confirmed and now - d.time <= GROUP_SECONDS]
        return done
