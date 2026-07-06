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


class FewShotLearner:
    """A REAL, dependency-free learner: it curates a few-shot exemplar bank.

    Not a stand-in — it does actual, useful work with no GPU: it maintains the
    top-N highest-reward interactions per domain as few-shot exemplars. Many
    production copilots improve precisely this way (better in-context examples),
    and NVIDIA's own blueprint runs an ICL/few-shot arm alongside LoRA. `answer`
    lets it act as a model: it returns the best exemplar's output for a domain,
    so it can be scored by a Judge and wired into win-rate / regression tests.
    """

    def __init__(self, bank_size: int = 8) -> None:
        self.bank_size = bank_size
        self._bank: dict[str, list[Interaction]] = {}
        self._trained = 0

    def quality(self) -> float:
        """Quality = mean exemplar reward × bank fill.

        Fill (how full the per-domain banks are, 0→1) makes quality RISE as the
        wheel accumulates good exemplars, then plateau — so the accelerometer
        sees real batch-over-batch improvement instead of an instantly-saturated
        mean. Honest: a half-full bank of great answers is worth less than a full
        one because it covers fewer situations.
        """
        allex = [i for lst in self._bank.values() for i in lst]
        if not allex:
            return 0.0
        mean_reward = sum((i.reward_score or 0.0) for i in allex) / len(allex)
        capacity = max(1, len(self._bank) * self.bank_size)
        fill = min(1.0, len(allex) / capacity)
        return round(mean_reward * fill, 4)

    def train(self, batch: list[Interaction]) -> float:
        for it in batch:
            dom = it.domain or "_"
            lst = self._bank.setdefault(dom, [])
            lst.append(it)
            lst.sort(key=lambda i: -(i.reward_score or 0.0))
            del lst[self.bank_size :]           # keep only the top-N per domain
        self._trained += len(batch)
        return self.quality()

    def answer(self, domain: str) -> str:
        """Best-known exemplar output for a domain (usable by a Judge)."""
        lst = self._bank.get(domain) or self._bank.get("_") or []
        return lst[0].output_text if lst else ""

    def exemplars(self, domain: str) -> list[Interaction]:
        return list(self._bank.get(domain, []))
