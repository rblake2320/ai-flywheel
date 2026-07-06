"""
The accelerometer — the metric that tells a spinning flywheel from a decorative one.

A flywheel isn't "working" just because data flows. It works only if each new
training batch produces a model that is measurably better than the last, AND
that improvement is at least partly attributable to cross-tenant data. This
records per-batch quality and reports whether the loop is ACCELERATING,
STEADY, or STALLING.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BatchRecord:
    index: int
    mean_reward: float
    n_examples: int
    n_source_tenants: int


@dataclass
class Accelerometer:
    """Tracks whether model quality improves batch-over-batch."""

    _batches: list[BatchRecord] = field(default_factory=list)

    def record(self, mean_reward: float, n_examples: int, n_source_tenants: int) -> BatchRecord:
        rec = BatchRecord(
            index=len(self._batches),
            mean_reward=round(mean_reward, 4),
            n_examples=n_examples,
            n_source_tenants=n_source_tenants,
        )
        self._batches.append(rec)
        return rec

    def delta(self) -> float | None:
        """Quality change from the previous batch to the latest."""
        if len(self._batches) < 2:
            return None
        return round(self._batches[-1].mean_reward - self._batches[-2].mean_reward, 4)

    def status(self) -> str:
        d = self.delta()
        if d is None:
            return "WARMING_UP"
        if d > 0.001:
            return "ACCELERATING"
        if d < -0.001:
            return "STALLING"
        return "STEADY"

    def report(self) -> dict:
        latest = self._batches[-1] if self._batches else None
        return {
            "batches": len(self._batches),
            "status": self.status(),
            "delta": self.delta(),
            "latest_mean_reward": latest.mean_reward if latest else None,
            # networked = the latest gains drew on >1 tenant's data
            "networked": bool(latest and latest.n_source_tenants >= 2),
        }
