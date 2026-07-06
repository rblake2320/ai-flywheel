"""
Per-tenant lift attribution — the network-effect proof and the pricing lever.

The whole promise is that each tenant gets MORE out of the shared engine than it
puts in. This makes that measurable per tenant: how much a tenant CONTRIBUTES
(accepted, shared learnings) vs how much it GAINS (learnings it receives from
others). The ratio is both the proof the flywheel is worth joining and the
natural basis for incentive/pricing design.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class LiftLedger:
    """Tracks contribution vs gain per tenant."""

    _contributed: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _received: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_contribution(self, tenant_id: str, n: int = 1) -> None:
        self._contributed[tenant_id] += n

    def record_gain(self, tenant_id: str, n: int) -> None:
        self._received[tenant_id] += n

    def lift(self, tenant_id: str) -> dict:
        c = self._contributed.get(tenant_id, 0)
        g = self._received.get(tenant_id, 0)
        # lift ratio: learnings received per learning contributed (>1 = net winner)
        ratio = (g / c) if c else (float(g) if g else 0.0)
        return {"contributed": c, "gained": g, "lift_ratio": round(ratio, 3)}

    def report(self) -> dict:
        tenants = set(self._contributed) | set(self._received)
        return {t: self.lift(t) for t in sorted(tenants)}
