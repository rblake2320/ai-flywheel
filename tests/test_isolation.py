"""Tenant boundary: proprietary content must NEVER cross into the shared engine."""
import pytest

from aiflywheel.core.interaction import Interaction
from aiflywheel.tenancy.tenant import IsolationError, IsolationGuard


def _mk(**kw):
    base = dict(id="i1", tenant_id="mk-copilot", timestamp="t")
    base.update(kw)
    return Interaction(**base)


def test_shared_record_excludes_private_text():
    it = _mk(
        input_text="Mary Kay seller SECRET comp plan question",
        output_text="proprietary answer",
        system_prompt="MK private prompt",
        reward_score=0.9,
        domain="retail",
        cross_learning="objection pattern lifts conversion",
    )
    shared = it.to_shared()
    blob = repr(shared)
    assert "SECRET" not in blob
    assert "proprietary answer" not in blob
    assert "MK private prompt" not in blob
    # but the anonymized signal DID cross
    assert shared["reward_score"] == 0.9
    assert shared["cross_learning"] == "objection pattern lifts conversion"
    assert shared["domain"] == "retail"


def test_guard_passes_clean_record():
    it = _mk(input_text="private", reward_score=0.8, cross_learning="lesson")
    assert IsolationGuard().check_outbound(it) is not None


def test_guard_blocks_private_leak(monkeypatch):
    it = _mk(input_text="leaky-secret", reward_score=0.8)

    # simulate a buggy redactor that lets private text through
    def bad_to_shared():
        return {"id": "i1", "tenant_id": "mk-copilot", "note": "leaky-secret"}

    monkeypatch.setattr(it, "to_shared", bad_to_shared)
    with pytest.raises(IsolationError):
        IsolationGuard().check_outbound(it)


def test_has_private_leak_detects_content():
    it = _mk(input_text="topsecret")
    assert it.has_private_leak({"x": "contains topsecret here"}) is True
    assert it.has_private_leak({"x": "clean"}) is False
