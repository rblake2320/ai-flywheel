"""
EventLogSource — the brain's event log feeds the flywheel.

Architectural rule: the flywheel does NOT open its own capture path — that would
recreate the fragmentation this whole effort exists to end. The brain's
append-only event log is the single honest source of experience. This adapter
tails that log (JSONL) and maps events to `Interaction`s the engine can train on,
tracking a byte offset so each event is consumed exactly once across restarts.

The engine stays the learning organ: it reads the brain's events, it does not
own them. Mapping is intentionally minimal and explicit — only fields the brain
already emits become flywheel signal; nothing is invented.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from aiflywheel.core.interaction import Interaction


def event_to_interaction(ev: dict) -> Interaction | None:
    """Map ONE brain event to an Interaction, or None if it's not trainable.

    Trainable = has a prompt/response pair and a tenant. Reward comes from the
    brain's own signal (explicit feedback, task success); provenance defaults to
    'real' since these are real events, not model-generated.
    """
    prompt = ev.get("input") or ev.get("prompt") or ev.get("query") or ""
    response = ev.get("output") or ev.get("response") or ev.get("answer") or ""
    tenant = ev.get("tenant_id") or ev.get("source") or ev.get("agent")
    if not (prompt and response and tenant):
        return None
    return Interaction(
        id=str(ev.get("id", "")),
        tenant_id=str(tenant),
        timestamp=str(ev.get("timestamp", "")),
        input_text=str(prompt),
        output_text=str(response),
        reward_score=ev.get("reward"),
        user_feedback=ev.get("feedback"),
        domain=ev.get("domain"),
        cross_learning=ev.get("learning") or ev.get("cross_learning"),
        provenance=str(ev.get("provenance", "real")),
    )


@dataclass
class EventLogSource:
    """Incrementally tails a brain event-log JSONL, yielding Interactions."""

    path: str
    _offset: int = 0
    _partial: str = field(default="", repr=False)

    def read_new(self) -> list[Interaction]:
        """Return Interactions for events appended since the last read.

        Offset-tracked and partial-line tolerant, so a concurrent writer can't
        make us drop or double-count an event across polls/restarts.
        """
        p = Path(self.path)
        if not p.exists():
            return []
        out: list[Interaction] = []
        with p.open("r", encoding="utf-8") as f:
            f.seek(self._offset)
            chunk = f.read()
            self._offset = f.tell()
        data = self._partial + chunk
        lines = data.split("\n")
        self._partial = lines.pop()  # last item is an incomplete line (or "")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                it = event_to_interaction(json.loads(line))
            except Exception:  # noqa: BLE001 - one bad line must not stop the tail
                continue
            if it is not None:
                out.append(it)
        return out


def feed_engine(engine, source: EventLogSource) -> int:
    """Pump all new events from the source into the engine. Returns count fed."""
    fed = 0
    for it in source.read_new():
        try:
            engine.submit(it)
            fed += 1
        except KeyError:
            # unknown tenant — the brain emitted a source not registered as a
            # tenant yet. Skip rather than crash; registration is an explicit act.
            continue
    return fed
