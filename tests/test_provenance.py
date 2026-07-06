"""Model-collapse defense: the real-data floor keeps batches from going synthetic."""
from aiflywheel.contract.sdk import FlywheelClient
from aiflywheel.core.interaction import Interaction
from aiflywheel.core.provenance import RealDataFloor, real_fraction
from aiflywheel.engine import FlywheelEngine
from aiflywheel.tenancy.tenant import Tenant


def _it(prov):
    return Interaction(id="x", tenant_id="t", timestamp="", reward_score=0.9, provenance=prov)


def test_real_fraction():
    batch = [_it("real"), _it("real"), _it("synthetic"), _it("model")]
    assert real_fraction(batch) == 0.5


def test_floor_passes_safe_batch():
    f = RealDataFloor(min_real_fraction=0.5)
    assert f.check([_it("real"), _it("real"), _it("synthetic")]) is True


def test_floor_trims_synthetic_heavy_batch():
    f = RealDataFloor(min_real_fraction=0.5)
    batch = [_it("real")] + [_it("synthetic")] * 9
    kept = f.enforce(batch)
    assert real_fraction(kept) >= 0.5
    assert len(kept) < len(batch)


def test_floor_refuses_all_synthetic():
    f = RealDataFloor(min_real_fraction=0.5)
    assert f.enforce([_it("synthetic"), _it("model")]) == []


def test_engine_refuses_all_synthetic_batch():
    eng = FlywheelEngine(batch_size=10, floor=RealDataFloor(min_real_fraction=0.5))
    eng.add_tenant(Tenant("t", domain="d"))
    c = FlywheelClient(eng, "t")
    for _ in range(10):
        c.report(input_text="p", reward=0.9, provenance="synthetic")
    # batch filled but was all-synthetic → no training happened
    assert eng.accel.report()["batches"] == 0
    assert eng.learner.quality() == 0.5   # unchanged


def test_provenance_crosses_but_content_does_not():
    it = Interaction(id="i", tenant_id="t", timestamp="", input_text="secret",
                     reward_score=0.9, provenance="synthetic")
    shared = it.to_shared()
    assert shared["provenance"] == "synthetic"
    assert "secret" not in repr(shared)
