"""
End-to-end proof the flywheel actually turns AND that the network effect is real.

The decisive test: a multi-tenant run must reach higher model quality than a
single-tenant run given the SAME volume of equally-good data — because the
cross-tenant network multiplier compounds. If that inequality ever fails, the
"Disney" part of the flywheel is not actually doing anything.
"""
from aiflywheel.contract.sdk import FlywheelClient
from aiflywheel.core.learner import SimulatedLearner
from aiflywheel.engine import FlywheelEngine
from aiflywheel.tenancy.tenant import Tenant


def _run(tenant_ids, total_interactions, batch_size=25):
    eng = FlywheelEngine(batch_size=batch_size, learner=SimulatedLearner(start=0.5))
    clients = []
    for tid in tenant_ids:
        eng.add_tenant(Tenant(tenant_id=tid, domain=tid.split("-")[0]))
        clients.append(FlywheelClient(eng, tid))
    for n in range(total_interactions):
        c = clients[n % len(clients)]
        c.report(
            input_text=f"private-{n}",
            output_text=f"answer-{n}",
            reward=0.9,
            domain=c.tenant_id.split("-")[0],
            cross_learning=f"lesson-{n}",
        )
    eng.flush()
    return eng


def test_flywheel_accelerates():
    eng = _run(["retail-a"], total_interactions=200)
    rep = eng.accel.report()
    assert rep["batches"] >= 2
    # quality rose over the run
    assert eng.learner.quality() > 0.5
    assert rep["status"] in {"ACCELERATING", "STEADY"}


def test_network_effect_beats_single_tenant():
    single = _run(["retail-a"], total_interactions=300)
    multi = _run(["retail-a", "travel-b", "realty-c"], total_interactions=300)
    # same data volume + reward; multi-tenant must end up SMARTER
    assert multi.learner.quality() > single.learner.quality()


def test_hub_is_networked_with_multiple_tenants():
    eng = _run(["retail-a", "travel-b"], total_interactions=100)
    cov = eng.hub.coverage()
    assert cov["is_networked"] is True
    assert cov["contributing_tenants"] >= 2


def test_hub_not_networked_single_tenant():
    eng = _run(["retail-a"], total_interactions=100)
    assert eng.hub.coverage()["is_networked"] is False


def test_consumer_pulls_only_others_learnings():
    eng = _run(["retail-a", "travel-b"], total_interactions=100)
    pulled = eng.pull("retail-a")["cross_learnings"]
    # retail-a should receive travel-b's learnings, none of its own
    assert all(ln.source_tenant != "retail-a" for ln in pulled)
    assert any(ln.source_tenant == "travel-b" for ln in pulled)


def test_unknown_tenant_rejected():
    eng = FlywheelEngine()
    c = FlywheelClient(eng, "ghost")
    try:
        c.report(reward=0.9)
        raise AssertionError("should have rejected unknown tenant")
    except KeyError:
        pass


def test_health_reports_turning_state():
    eng = _run(["retail-a", "travel-b"], total_interactions=120)
    h = eng.health()
    assert h["tenants"] == 2
    assert h["hub"]["is_networked"] is True
    assert h["model_quality"] > 0.5
