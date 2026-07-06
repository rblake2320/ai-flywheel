"""
Multi-tenancy: the boundary between a vertical copilot and the shared engine.

A Tenant is a vertical business copilot (MK Copilot, a real-estate copilot, a
car-sales copilot...). Each is isolated: its private knowledge stays local, it
contributes only anonymized learnings upward, and it receives shared model
improvements downward. The engine is business-agnostic; tenants are the
business-specific plug-ins.

The IsolationGuard enforces "what must never cross" — it inspects every
boundary-crossing payload and refuses any that carries a tenant's private
content. Fail closed: on any doubt, block.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aiflywheel.core.interaction import Interaction


class IsolationError(RuntimeError):
    """Raised when a payload would leak tenant-private content across the boundary."""


@dataclass
class Tenant:
    """A registered vertical copilot plugged into the flywheel."""

    tenant_id: str
    domain: str                       # generic vertical, e.g. "retail", "real_estate"
    display_name: str = ""
    # opt-in: does this tenant contribute learnings to the shared hub?
    contributes: bool = True
    # opt-in: does this tenant consume shared improvements?
    consumes: bool = True
    tags: list[str] = field(default_factory=list)


class IsolationGuard:
    """Enforces the tenant boundary on every crossing."""

    def check_outbound(self, interaction: Interaction) -> dict:
        """Validate + produce the redacted shared record for one interaction.

        Raises IsolationError if the redaction would still carry private content.
        """
        shared = interaction.to_shared()
        if interaction.has_private_leak(shared):
            raise IsolationError(
                f"private content leaked in shared record for {interaction.id}"
            )
        # tenant_id may cross (it's an opaque handle), but raw proprietary text
        # must never appear as a value.
        for key in ("input_text", "output_text", "system_prompt"):
            if key in shared:
                raise IsolationError(f"forbidden private field '{key}' in shared record")
        return shared


class TenantRegistry:
    """Registry of all tenants plugged into the engine."""

    def __init__(self) -> None:
        self._tenants: dict[str, Tenant] = {}

    def register(self, tenant: Tenant) -> Tenant:
        if tenant.tenant_id in self._tenants:
            raise ValueError(f"tenant already registered: {tenant.tenant_id}")
        self._tenants[tenant.tenant_id] = tenant
        return tenant

    def get(self, tenant_id: str) -> Tenant:
        if tenant_id not in self._tenants:
            raise KeyError(f"unknown tenant: {tenant_id}")
        return self._tenants[tenant_id]

    def all(self) -> list[Tenant]:
        return list(self._tenants.values())

    def contributors(self) -> list[Tenant]:
        return [t for t in self._tenants.values() if t.contributes]

    def consumers(self) -> list[Tenant]:
        return [t for t in self._tenants.values() if t.consumes]
