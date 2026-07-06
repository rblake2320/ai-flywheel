"""
The tenant plug-in SDK — how a vertical copilot talks to the flywheel.

This is the whole public contract a business copilot needs. It hides the engine
internals and enforces that a tenant can only ever act as ITSELF (its
tenant_id is bound at construction, so one tenant can't submit as another).

    client = FlywheelClient(engine, tenant_id="mk-copilot")
    client.report(input_text=..., output_text=..., reward=0.9,
                  cross_learning="objection-handling pattern X lifts conversion")
    update = client.sync()        # pull improved model + others' learnings
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from aiflywheel.core.interaction import Interaction
from aiflywheel.engine import FlywheelEngine, SubmitResult

_counter = itertools.count(1)


@dataclass
class FlywheelClient:
    """A single tenant's handle to the shared engine. Bound to one tenant_id."""

    engine: FlywheelEngine
    tenant_id: str

    def report(
        self,
        *,
        input_text: str = "",
        output_text: str = "",
        system_prompt: str = "",
        reward: float | None = None,
        feedback: int | None = None,
        domain: str | None = None,
        cross_learning: str | None = None,
        timestamp: str = "",
        tags: list[str] | None = None,
    ) -> SubmitResult:
        """Report one interaction. Private text stays local; only signal crosses."""
        interaction = Interaction(
            id=f"{self.tenant_id}-{next(_counter)}",
            tenant_id=self.tenant_id,        # bound identity — cannot spoof another tenant
            timestamp=timestamp,
            input_text=input_text,
            output_text=output_text,
            system_prompt=system_prompt,
            reward_score=reward,
            user_feedback=feedback,
            domain=domain,
            cross_learning=cross_learning,
            tags=tags or [],
        )
        return self.engine.submit(interaction)

    def sync(self) -> dict:
        """Pull the improved model + anonymized learnings from all other tenants."""
        return self.engine.pull(self.tenant_id)
