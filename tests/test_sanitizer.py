"""The cross_learning leak is closed: proprietary content can't ride the shared field."""
from aiflywheel.contract.sdk import FlywheelClient
from aiflywheel.engine import FlywheelEngine
from aiflywheel.safety.sanitizer import LearningSanitizer
from aiflywheel.tenancy.tenant import Tenant


def test_redacts_email_phone_currency_digits():
    s = LearningSanitizer()
    r = s.sanitize("pattern works, contact jane@mk.com or 555-123-4567, worth $12,500 acct 998877")
    assert r.ok
    assert "jane@mk.com" not in r.text
    assert "555-123-4567" not in r.text
    assert "$12,500" not in r.text
    assert "998877" not in r.text
    assert "[REDACTED]" in r.text


def test_secret_terms_redacted():
    s = LearningSanitizer(secret_terms=frozenset({"Project Aurora"}))
    r = s.sanitize("the Project Aurora tactic lifts retention")
    assert r.ok
    assert "Project Aurora" not in r.text


def test_hard_block_rejects():
    s = LearningSanitizer(hard_block=frozenset({"comp plan"}))
    r = s.sanitize("the comp plan detail is X")
    assert r.ok is False
    assert r.reason.startswith("hard_block")


def test_empty_after_scrub_rejected():
    s = LearningSanitizer()
    r = s.sanitize("jane@mk.com 5551234567 $99")
    assert r.ok is False
    assert r.reason == "empty_after_scrub"


def test_length_capped():
    s = LearningSanitizer(max_length=20)
    r = s.sanitize("x" * 100)
    assert len(r.text) <= 20


def test_engine_blocks_unsafe_learning_from_hub():
    eng = FlywheelEngine(batch_size=1000)
    eng.add_tenant(Tenant("mk-copilot", domain="retail", tags=[]))
    # engine with a hard-block sanitizer
    eng.sanitizer = LearningSanitizer(hard_block=frozenset({"secret comp plan"}))
    c = FlywheelClient(eng, "mk-copilot")
    res = c.report(input_text="private", reward=0.9,
                   cross_learning="the secret comp plan is 40 percent")
    assert res.accepted is True
    assert res.shared is False              # learning did NOT cross
    assert res.reject_reason.startswith("hard_block")
    assert eng.hub.coverage()["total_learnings"] == 0
