"""
Reward validation — stop trusting tenant-supplied scores blindly.

The adaptive valve is only as trustworthy as the reward feeding it. A buggy or
self-serving tenant reporting reward=1.0 on everything would poison the shared
training set. This module:

  - clamps/validates the raw reward into [0,1] (rejecting NaN/None/out-of-range),
  - lets the engine attach an optional RewardVerifier that can DOWN-weight or
    veto a tenant's self-reported score (e.g. an independent judge model),
  - tracks per-tenant reporting stats so graders can spot a tenant whose
    self-reported rewards are implausibly high (a trust signal).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class RewardVerifier(Protocol):
    """An independent check on a tenant's self-reported reward."""

    def verify(self, tenant_id: str, reported: float) -> float:
        """Return an adjusted reward in [0,1] (may down-weight or veto)."""
        ...


def clamp_reward(value: float | None) -> float | None:
    """Coerce a raw reward to a valid [0,1] float, or None if unusable."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return max(0.0, min(1.0, v))


@dataclass
class RewardTracker:
    """Per-tenant reward stats — a lightweight trust signal."""

    _sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _n: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, tenant_id: str, reward: float) -> None:
        self._sum[tenant_id] += reward
        self._n[tenant_id] += 1

    def mean(self, tenant_id: str) -> float:
        n = self._n[tenant_id]
        return self._sum[tenant_id] / n if n else 0.0

    def suspicious(self, tenant_id: str, ceiling: float = 0.97, min_n: int = 20) -> bool:
        """True if a tenant's mean self-reported reward is implausibly high."""
        return self._n[tenant_id] >= min_n and self.mean(tenant_id) >= ceiling
