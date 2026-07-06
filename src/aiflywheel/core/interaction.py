"""
The atomic unit of the flywheel: one tenant interaction.

The single most important idea in this file is the TENANT BOUNDARY. An
Interaction carries two kinds of data:

  - PRIVATE fields  — proprietary to the tenant (a vertical copilot's raw
    input/output, its customer's data, its private knowledge). These NEVER
    leave the tenant boundary and NEVER enter shared training.
  - SHAREABLE fields — anonymized, generalized learnings (a reward signal, a
    distilled `cross_learning`) that are safe to contribute to the shared hub
    so every other tenant gets smarter.

`to_shared()` is the ONLY sanctioned way data crosses from a tenant into the
shared engine. It emits a redacted record containing shareable fields only.
If a field isn't explicitly shareable, it does not cross. Fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Interaction:
    """A single interaction produced by a tenant's vertical copilot."""

    # --- identity (shareable: opaque ids, no proprietary content) ---
    id: str
    tenant_id: str
    timestamp: str

    # --- PRIVATE: proprietary tenant content — never crosses the boundary ---
    input_text: str = ""
    output_text: str = ""
    system_prompt: str = ""
    private_metadata: dict[str, Any] = field(default_factory=dict)

    # --- SHAREABLE: anonymized signal safe for the shared hub ---
    reward_score: float | None = None          # 0.0–1.0 quality, no content
    user_feedback: int | None = None           # -1 / 0 / 1
    domain: str | None = None                   # generic vertical, e.g. "retail"
    cross_learning: str | None = None           # distilled, anonymized lesson
    tags: list[str] = field(default_factory=list)
    # provenance is the #1 defense against model collapse: the engine must know
    # whether an example is real human/production data or model-generated, so it
    # can ACCUMULATE (keep a real-data floor) rather than REPLACE. Never train a
    # batch that is all synthetic. See core/provenance.py.
    provenance: str = "real"                    # real | human | synthetic | model

    # fields that may EVER leave a tenant — the allowlist. Anything not here
    # is private by construction.
    _SHAREABLE = frozenset(
        {"id", "tenant_id", "timestamp", "reward_score", "user_feedback",
         "domain", "cross_learning", "tags", "provenance"}
    )

    def to_shared(self) -> dict[str, Any]:
        """Redacted, boundary-crossing view — shareable fields ONLY.

        This is the only path from tenant-private to the shared engine.
        """
        out: dict[str, Any] = {}
        for name in self._SHAREABLE:
            val = getattr(self, name)
            if val is not None and val != [] and val != {}:
                out[name] = val
        return out

    def has_private_leak(self, blob: dict[str, Any]) -> bool:
        """True if any of THIS interaction's private content appears in a VALUE
        of `blob`.

        Checks values only — structural key names (e.g. "timestamp") are not
        leaked content, so matching against them would be a false positive.
        Used by the isolation guard to assert nothing proprietary escaped.
        """
        needles = [self.input_text, self.output_text, self.system_prompt]
        needles = [n for n in needles if n]
        if not needles:
            return False
        hay = " \x00 ".join(str(v) for v in blob.values())
        return any(n in hay for n in needles)
