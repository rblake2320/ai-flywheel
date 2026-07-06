"""v0.6.0: self-reflection — rollbacks become remembered, recallable WhyCases."""
import json

from aiflywheel.contract.sdk import FlywheelClient
from aiflywheel.engine import FlywheelEngine
from aiflywheel.metrics.promotion import PromotionGate
from aiflywheel.reflection.whycase import WhyStore
from aiflywheel.tenancy.tenant import Tenant


def test_whycase_written_in_engine_schema(tmp_path):
    store = WhyStore(outbox_dir=str(tmp_path))
    path = store.record(
        title="training regression",
        root_cause="batch from tenant mk-copilot lowered quality 0.8->0.5",
        why_not_caught="passed intake valve but degraded model",
        prevent_next_time="raise scrutiny for retail batches",
        generalizable_pattern="regression domains=['retail']",
        created_at="2026-07-06T00:00:00Z",
    )
    case = json.loads(open(path, encoding="utf-8").read())
    for req in ("caseId", "idempotencyKey", "createdAt", "title", "rootCause",
                "whyNotCaught", "whyFixWorked", "preventNextTime",
                "sensitivity", "secretScanResult"):
        assert req in case
    assert case["secretScanResult"]["clean"] is True


def test_secret_is_redacted_before_write(tmp_path):
    store = WhyStore(outbox_dir=str(tmp_path))
    path = store.record(
        title="leak test ghp_ABCDEFGHIJKLMNOPQRSTUVWX123456",
        root_cause="token ghp_ABCDEFGHIJKLMNOPQRSTUVWX123456 appeared",
        why_not_caught="x", prevent_next_time="y",
    )
    case = json.loads(open(path, encoding="utf-8").read())
    assert "ghp_ABCDEFGHIJKLMNOPQRSTUVWX123456" not in json.dumps(case)
    assert case["secretScanResult"]["clean"] is False
    assert case["secretScanResult"]["secretsFound"] >= 1


def test_truth_check_rejects_fabricated_regression(tmp_path):
    # a "regression" case whose real numbers show improvement is refused —
    # the fact layer prevents filling the store with untrue cases.
    store = WhyStore(outbox_dir=str(tmp_path))
    import pytest
    with pytest.raises(ValueError, match="isn't real"):
        store.record(
            title="fake regression", root_cause="claims a drop that didn't happen",
            why_not_caught="x", prevent_next_time="y",
            measured_facts={"prev_quality": 0.5, "new_quality": 0.8},  # improved!
        )
    assert store.count() == 0


def test_measured_facts_are_recorded(tmp_path):
    store = WhyStore(outbox_dir=str(tmp_path))
    path = store.record(
        title="real regression", root_cause="quality dropped",
        why_not_caught="x", prevent_next_time="y",
        measured_facts={"prev_quality": 0.8, "new_quality": 0.5, "batch_size": 5},
    )
    case = json.loads(open(path, encoding="utf-8").read())
    assert case["measuredFacts"]["prev_quality"] == 0.8
    assert case["measuredFacts"]["new_quality"] == 0.5


def test_recall_finds_prior_regression(tmp_path):
    store = WhyStore(outbox_dir=str(tmp_path))
    store.record(title="regression in retail", root_cause="retail batch degraded model",
                 why_not_caught="x", prevent_next_time="watch retail",
                 generalizable_pattern="regression domains retail")
    hits = store.recall("retail batch regression")
    assert hits and "retail" in hits[0]["rootCause"]
    assert store.seen_before("retail batch regression") is True
    assert store.seen_before("completely unrelated aerospace telemetry") is False


def test_recall_survives_restart(tmp_path):
    WhyStore(outbox_dir=str(tmp_path)).record(
        title="regression in autos", root_cause="automotive batch degraded",
        why_not_caught="x", prevent_next_time="y",
        generalizable_pattern="regression domains automotive")
    # a fresh store loads existing cases from the outbox
    reloaded = WhyStore(outbox_dir=str(tmp_path))
    assert reloaded.count() == 1
    assert reloaded.seen_before("automotive regression") is True


def test_engine_records_whycase_on_rollback(tmp_path):
    class DegradingLearner:
        def __init__(self):
            self._q = 0.8
        def quality(self):
            return self._q
        def train(self, batch):
            self._q = 0.3
            return self._q
        def snapshot(self):
            return self._q
        def rollback(self, snap):
            self._q = snap

    store = WhyStore(outbox_dir=str(tmp_path))
    eng = FlywheelEngine(batch_size=5, learner=DegradingLearner(),
                         promotion=PromotionGate(), reflector=store)
    eng.add_tenant(Tenant("mk-copilot", domain="retail"))
    c = FlywheelClient(eng, "mk-copilot")
    for n in range(5):
        c.report(input_text=f"seller question number {n}",
                 output_text=f"a helpful retail answer {n}", reward=0.9, domain="retail")
    h = eng.health()
    assert h["rollbacks"] == 1
    assert h["why_cases"] == 1                       # the rollback was remembered
    assert store.seen_before("retail regression mk-copilot") is True
