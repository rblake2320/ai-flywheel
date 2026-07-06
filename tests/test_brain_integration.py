"""v0.7.0: brain-integration seams — event-log source + pluggable recall.

The flywheel is the learning organ: it CONSUMES the brain's event log and
becomes a CLIENT of the brain's /recall. It never opens a second capture path
or a second memory. These tests prove the seams without any brain present.
"""
import json

from aiflywheel.contract.sdk import FlywheelClient  # noqa: F401 (kept for parity)
from aiflywheel.engine import FlywheelEngine
from aiflywheel.ingest.event_log import EventLogSource, event_to_interaction, feed_engine
from aiflywheel.reflection.recall import LocalRecall, RecallProvider, RemoteRecall
from aiflywheel.reflection.whycase import WhyStore
from aiflywheel.tenancy.tenant import Tenant


# --- event log source ---
def test_event_maps_to_interaction():
    it = event_to_interaction({
        "id": "e1", "tenant_id": "blake-os", "input": "recall my deadlines",
        "output": "BPC patent due 2027-04-03", "reward": 0.9, "domain": "memory",
    })
    assert it is not None
    assert it.tenant_id == "blake-os"
    assert it.reward_score == 0.9
    assert it.provenance == "real"


def test_non_trainable_event_is_skipped():
    assert event_to_interaction({"id": "x", "note": "no prompt/response"}) is None


def test_event_log_incremental_read(tmp_path):
    log = tmp_path / "events.jsonl"
    log.write_text(json.dumps({"id": "1", "tenant_id": "t", "input": "q1",
                               "output": "a1", "reward": 0.9}) + "\n", encoding="utf-8")
    src = EventLogSource(path=str(log))
    first = src.read_new()
    assert len(first) == 1
    # nothing new yet
    assert src.read_new() == []
    # append another event → only the new one is read
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "2", "tenant_id": "t", "input": "q2",
                            "output": "a2", "reward": 0.8}) + "\n")
    second = src.read_new()
    assert len(second) == 1 and second[0].id == "2"


def test_feed_engine_from_event_log(tmp_path):
    log = tmp_path / "events.jsonl"
    lines = [json.dumps({"id": str(n), "tenant_id": "blake-os", "input": f"q{n}",
                         "output": f"a{n}", "reward": 0.9, "domain": "memory"})
             for n in range(6)]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    eng = FlywheelEngine(batch_size=100)
    eng.add_tenant(Tenant("blake-os", domain="memory"))
    fed = feed_engine(eng, EventLogSource(path=str(log)))
    assert fed == 6


def test_feed_engine_skips_unregistered_tenant(tmp_path):
    log = tmp_path / "events.jsonl"
    log.write_text(json.dumps({"id": "1", "tenant_id": "ghost", "input": "q",
                               "output": "a", "reward": 0.9}) + "\n", encoding="utf-8")
    eng = FlywheelEngine()          # no tenants registered
    fed = feed_engine(eng, EventLogSource(path=str(log)))
    assert fed == 0                 # unknown tenant skipped, no crash


# --- pluggable recall ---
def test_local_recall_wraps_whystore(tmp_path):
    store = WhyStore(outbox_dir=str(tmp_path))
    store.record(title="regression in memory", root_cause="memory batch degraded",
                 why_not_caught="x", prevent_next_time="y",
                 generalizable_pattern="regression domains memory",
                 measured_facts={"prev_quality": 0.8, "new_quality": 0.5})
    rec = LocalRecall(store=store)
    assert isinstance(rec, RecallProvider)
    assert rec.seen_before("memory regression") is True


def test_remote_recall_fails_open_when_brain_down():
    # no brain at this port → recall returns empty / not-seen, never raises
    rec = RemoteRecall(base_url="http://127.0.0.1:59999", timeout=0.2)
    assert isinstance(rec, RecallProvider)
    assert rec.recall("anything") == []
    assert rec.seen_before("anything") is False


def test_engine_prefers_injected_recall(tmp_path):
    class AlwaysSeen:
        def seen_before(self, q):
            return True
        def recall(self, q, k=3):
            return [{"hit": True}]

    eng = FlywheelEngine(recall=AlwaysSeen())
    eng.add_tenant(Tenant("t", domain="memory"))
    from aiflywheel.core.interaction import Interaction
    batch = [Interaction(id="x", tenant_id="t", timestamp="", domain="memory")]
    assert eng.seen_regression_before(batch) is True
