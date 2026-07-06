"""
Persistence — make the flywheel durable across restarts.

The shared hub of learnings and the accelerometer's batch history are the
flywheel's accumulated momentum. Losing them on restart resets the wheel to
zero. This store snapshots the shareable, non-proprietary state to JSONL:

  - shared learnings (already anonymized + sanitized before they got here),
  - accelerometer batch records (aggregate metrics only).

It deliberately does NOT persist raw interactions or anything tenant-private —
that data never reaches the engine in the first place, and never should touch
disk here. Load rehydrates the hub and accelerometer so the wheel keeps its
momentum.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from aiflywheel.learning.hub import CrossLearningHub, SharedLearning
from aiflywheel.metrics.accelerometer import Accelerometer, BatchRecord


class FlywheelStore:
    """JSONL snapshot/restore of the flywheel's durable, shareable momentum."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, hub: CrossLearningHub, accel: Accelerometer) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for ln in hub._learnings:            # already sanitized/anonymized
                f.write(json.dumps({"t": "learning", **asdict(ln)}) + "\n")
            for b in accel._batches:
                f.write(json.dumps({"t": "batch", **asdict(b)}) + "\n")

    def load(self, hub: CrossLearningHub, accel: Accelerometer) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                kind = rec.pop("t", None)
                if kind == "learning":
                    hub.contribute(SharedLearning(**rec))
                elif kind == "batch":
                    accel._batches.append(BatchRecord(**rec))
