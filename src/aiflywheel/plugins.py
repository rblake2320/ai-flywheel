"""
Plugin discovery via entry points — makes the engine an extensible ecosystem.

Third-party packages register real backends (a TRL Learner, a DeepEval Judge, a
SemHash dedup stage) under the `ai_flywheel.plugins` entry-point group. This
lets `pip install ai-flywheel-trl` add a trainer with zero code changes here.

Uses the keyword form `entry_points(group=...)` — the dict-subscript form was
removed in Python 3.12, so this is the portable call across 3.10–3.13+.
"""
from __future__ import annotations

from importlib.metadata import entry_points

GROUP = "ai_flywheel.plugins"


def discover() -> dict[str, object]:
    """Return {name: loaded_plugin} for everything registered under the group."""
    found: dict[str, object] = {}
    try:
        eps = entry_points(group=GROUP)
    except TypeError:  # pragma: no cover - very old importlib.metadata
        eps = entry_points().get(GROUP, [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            found[ep.name] = ep.load()
        except Exception:  # noqa: BLE001 - a bad plugin must not break discovery
            continue
    return found
