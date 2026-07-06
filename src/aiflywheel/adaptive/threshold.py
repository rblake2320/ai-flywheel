"""
Adaptive reward threshold — the valve that decides what the flywheel learns.

The old design hardcoded 0.8. Too high and the flywheel starves (little passes,
slow learning); too low and it learns mediocrity and degrades. This controller
floats the threshold to hold a target ACCEPT RATE: if too much is passing, it
tightens; if too little, it loosens. Bounded so it can never fully open or slam
shut. This keeps a steady, healthy intake regardless of absolute score drift.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveThreshold:
    """A self-tuning quality threshold targeting a desired accept rate."""

    value: float = 0.8
    target_accept_rate: float = 0.3   # aim to admit the top ~30% of interactions
    lower_bound: float = 0.5
    upper_bound: float = 0.95
    step: float = 0.02                # max move per update
    _seen: int = 0
    _accepted: int = 0

    def observe(self, score: float) -> bool:
        """Record a score, return whether it passes the current threshold."""
        passed = score >= self.value
        self._seen += 1
        if passed:
            self._accepted += 1
        return passed

    @property
    def accept_rate(self) -> float:
        return self._accepted / self._seen if self._seen else 0.0

    def update(self) -> float:
        """Nudge the threshold toward the target accept rate. Call per batch."""
        if self._seen == 0:
            return self.value
        rate = self.accept_rate
        error = rate - self.target_accept_rate
        # too much passing (error>0) → raise bar; too little → lower it.
        move = max(-self.step, min(self.step, error))
        self.value = round(
            min(self.upper_bound, max(self.lower_bound, self.value + move)), 4
        )
        self._seen = 0
        self._accepted = 0
        return self.value
