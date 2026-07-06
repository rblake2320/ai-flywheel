"""
The accelerometer — tells a spinning flywheel from a decorative one.

A flywheel isn't "working" just because data flows. Research on real systems
(2025-2026) says the naive signal — mean reward per batch — misses the two that
actually matter:

  - MARGINAL VALUE = Δquality / N-new-examples. If this trends to zero, the
    wheel is stalling and you're burning curation/compute for nothing. This is
    the true accelerometer.
  - DIVERSITY / COVERAGE = how many distinct tenants/domains fed the batch. If
    this falls while reward looks fine, you're heading into mode collapse — an
    early warning reward alone can't give.

We keep mean-reward and batch-delta too, but status now also degrades to
STALLING when marginal value collapses even if reward is flat.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BatchRecord:
    index: int
    mean_reward: float
    n_examples: int
    n_source_tenants: int
    model_quality: float = 0.0
    n_domains: int = 0
    real_fraction: float = 1.0


@dataclass
class Accelerometer:
    """Tracks whether model quality improves batch-over-batch, and why."""

    _batches: list[BatchRecord] = field(default_factory=list)

    def record(
        self,
        mean_reward: float,
        n_examples: int,
        n_source_tenants: int,
        model_quality: float = 0.0,
        n_domains: int = 0,
        real_fraction: float = 1.0,
    ) -> BatchRecord:
        rec = BatchRecord(
            index=len(self._batches),
            mean_reward=round(mean_reward, 4),
            n_examples=n_examples,
            n_source_tenants=n_source_tenants,
            model_quality=round(model_quality, 4),
            n_domains=n_domains,
            real_fraction=round(real_fraction, 4),
        )
        self._batches.append(rec)
        return rec

    def delta(self) -> float | None:
        """Model-quality change from the previous batch to the latest."""
        if len(self._batches) < 2:
            return None
        return round(self._batches[-1].model_quality - self._batches[-2].model_quality, 4)

    def marginal_value(self) -> float | None:
        """Δquality per new example — the true accelerometer. →0 means stalling."""
        d = self.delta()
        if d is None:
            return None
        n = self._batches[-1].n_examples or 1
        return round(d / n, 6)

    def status(self) -> str:
        d = self.delta()
        if d is None:
            return "WARMING_UP"
        mv = self.marginal_value() or 0.0
        if d > 0.001 and mv > 1e-5:
            return "ACCELERATING"
        if d < -0.001:
            return "STALLING"
        if mv <= 1e-6:
            return "STALLING"          # quality plateaued relative to data spent
        return "STEADY"

    def did_accelerate(self) -> bool:
        """True if quality ever rose batch-over-batch (a saturated wheel that
        climbed then plateaued still counts as having turned)."""
        return any(
            self._batches[i].model_quality > self._batches[i - 1].model_quality + 1e-6
            for i in range(1, len(self._batches))
        )

    def peak_quality(self) -> float | None:
        return max((b.model_quality for b in self._batches), default=None)

    def report(self) -> dict:
        latest = self._batches[-1] if self._batches else None
        return {
            "batches": len(self._batches),
            "status": self.status(),
            "did_accelerate": self.did_accelerate(),
            "peak_quality": self.peak_quality(),
            "delta": self.delta(),
            "marginal_value": self.marginal_value(),
            "latest_mean_reward": latest.mean_reward if latest else None,
            "latest_model_quality": latest.model_quality if latest else None,
            "diversity_tenants": latest.n_source_tenants if latest else 0,
            "diversity_domains": latest.n_domains if latest else 0,
            "real_fraction": latest.real_fraction if latest else None,
            # networked = the latest gains drew on >1 tenant's data
            "networked": bool(latest and latest.n_source_tenants >= 2),
        }
