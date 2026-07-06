"""
The Learner protocol — the pluggable "model training" step.

The engine is honest about its boundary: it does NOT ship a neural trainer.
It defines the CONTRACT a real trainer must satisfy (`train` on a batch,
`quality` of the current model) and provides a deterministic SimulatedLearner
so the whole loop runs, is testable, and demonstrates acceleration end-to-end
with no external dependencies. A production tenant supplies a real Learner
(SFT/DPO on the DGX Sparks, NVIDIA NeMo, etc.) behind the same interface.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from aiflywheel.core.interaction import Interaction


@runtime_checkable
class Learner(Protocol):
    """Anything that can improve a model from a batch of accepted interactions."""

    def train(self, batch: list[Interaction]) -> float:
        """Train on the batch; return the new model quality (0.0–1.0)."""
        ...

    def quality(self) -> float:
        """Current model quality (0.0–1.0)."""
        ...


class SimulatedLearner:
    """Deterministic stand-in trainer used for tests and demos.

    Models the real dynamic: quality rises with the volume and mean reward of
    accepted data, with diminishing returns, and rises FASTER when the batch
    draws on more distinct tenants (the network effect). No randomness — fully
    reproducible.
    """

    def __init__(self, start: float = 0.50, ceiling: float = 0.99) -> None:
        self._quality = start
        self._ceiling = ceiling

    def quality(self) -> float:
        return round(self._quality, 4)

    def train(self, batch: list[Interaction]) -> float:
        if not batch:
            return self.quality()
        mean_reward = sum((i.reward_score or 0.0) for i in batch) / len(batch)
        n_sources = len({i.tenant_id for i in batch})
        # network multiplier: more distinct tenants -> larger, compounding gain
        network_mult = 1.0 + 0.35 * (n_sources - 1)
        volume_factor = min(1.0, len(batch) / 50.0)
        headroom = self._ceiling - self._quality
        gain = headroom * 0.25 * mean_reward * volume_factor * network_mult
        self._quality = min(self._ceiling, self._quality + gain)
        return self.quality()
