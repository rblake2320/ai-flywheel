"""
Persistence — durable snapshot of ALL the flywheel's learned state.

Earlier this saved only the hub + accelerometer, so a restart silently dropped
the adaptive threshold's learned setpoint, the per-tenant trust stats, the
promotion/rollback counters, and the lift ledger — the wheel forgot most of what
it had learned about ITSELF. This snapshots every durable piece so "durable,
learns off everything" holds across restarts, not just for WhyCases.

Writes are atomic + fsync'd (temp → fsync → os.replace), matching the rigor the
WhyStore already has: a crash cannot leave a half-written or lost snapshot.

Persists only shareable/aggregate state — never raw interactions or tenant
-private content (that never reaches the engine, and never touches disk here).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from aiflywheel.learning.hub import SharedLearning
from aiflywheel.metrics.accelerometer import BatchRecord


class FlywheelStore:
    """Atomic, fsync'd JSON snapshot/restore of the engine's durable state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, engine) -> None:
        state = {
            "learnings": [asdict(ln) for ln in engine.hub._learnings],
            "batches": [asdict(b) for b in engine.accel._batches],
            "threshold": {
                "value": engine.threshold.value,
                "target_accept_rate": engine.threshold.target_accept_rate,
                "lower_bound": engine.threshold.lower_bound,
                "upper_bound": engine.threshold.upper_bound,
                "step": engine.threshold.step,
            },
            "counters": {
                "promotions": engine._promotions,
                "rollbacks": engine._rollbacks,
            },
            "lift": {
                "contributed": dict(engine.lift._contributed),
                "received": dict(engine.lift._received),
            },
            "rewards": {
                "sum": dict(engine.rewards._sum),
                "n": dict(engine.rewards._n),
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(state, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def load(self, engine) -> None:
        if not self.path.exists():
            return
        state = json.loads(self.path.read_text(encoding="utf-8"))

        for rec in state.get("learnings", []):
            engine.hub.contribute(SharedLearning(**rec))
        for rec in state.get("batches", []):
            engine.accel._batches.append(BatchRecord(**rec))

        th = state.get("threshold", {})
        for k, v in th.items():
            setattr(engine.threshold, k, v)

        c = state.get("counters", {})
        engine._promotions = c.get("promotions", 0)
        engine._rollbacks = c.get("rollbacks", 0)

        lift = state.get("lift", {})
        engine.lift._contributed.update(lift.get("contributed", {}))
        engine.lift._received.update(lift.get("received", {}))

        rw = state.get("rewards", {})
        engine.rewards._sum.update(rw.get("sum", {}))
        engine.rewards._n.update(rw.get("n", {}))
