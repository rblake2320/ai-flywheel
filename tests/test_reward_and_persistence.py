"""Reward is validated (not blindly trusted) and the wheel survives a restart."""
from aiflywheel.contract.sdk import FlywheelClient
from aiflywheel.core.reward import RewardTracker, clamp_reward
from aiflywheel.engine import FlywheelEngine
from aiflywheel.tenancy.tenant import Tenant


def test_clamp_reward():
    assert clamp_reward(1.5) == 1.0
    assert clamp_reward(-2) == 0.0
    assert clamp_reward(None) is None
    assert clamp_reward(float("nan")) is None
    assert clamp_reward("bad") is None
    assert clamp_reward(0.42) == 0.42


def test_engine_clamps_out_of_range_reward():
    eng = FlywheelEngine(batch_size=1000)
    eng.add_tenant(Tenant("t", domain="d"))
    res = FlywheelClient(eng, "t").report(input_text="p", reward=9.9)
    assert res.reward == 1.0                 # clamped, not 9.9


def test_verifier_can_downweight():
    class HalfVerifier:
        def verify(self, tenant_id, reported):
            return reported * 0.5

    eng = FlywheelEngine(batch_size=1000, verifier=HalfVerifier())
    eng.add_tenant(Tenant("t", domain="d"))
    res = FlywheelClient(eng, "t").report(input_text="p", reward=0.8)
    assert res.reward == 0.4


def test_reward_tracker_flags_suspicious():
    rt = RewardTracker()
    for _ in range(25):
        rt.record("greedy", 1.0)
    for _ in range(25):
        rt.record("honest", 0.6)
    assert rt.suspicious("greedy") is True
    assert rt.suspicious("honest") is False


def test_persistence_round_trip(tmp_path):
    path = str(tmp_path / "flywheel.jsonl")

    eng = FlywheelEngine(batch_size=10)
    eng.add_tenant(Tenant("retail-a", domain="retail"))
    eng.add_tenant(Tenant("travel-b", domain="travel"))
    ca, cb = FlywheelClient(eng, "retail-a"), FlywheelClient(eng, "travel-b")
    for n in range(40):
        (ca if n % 2 else cb).report(
            input_text=f"p{n}", reward=0.9, cross_learning=f"lesson {n}"
        )
    eng.flush()
    before = eng.hub.coverage()["total_learnings"]
    batches_before = eng.accel.report()["batches"]
    assert before > 0 and batches_before > 0
    eng.save(path)

    # simulate a restart: brand-new engine, load momentum
    eng2 = FlywheelEngine()
    eng2.load(path)
    assert eng2.hub.coverage()["total_learnings"] == before
    assert eng2.accel.report()["batches"] == batches_before
    assert eng2.hub.coverage()["is_networked"] is True


def test_persistence_covers_all_learned_state(tmp_path):
    from aiflywheel.metrics.promotion import PromotionGate
    path = str(tmp_path / "state.json")
    eng = FlywheelEngine(batch_size=10, promotion=PromotionGate())
    eng.add_tenant(Tenant("retail-a", domain="retail"))
    eng.add_tenant(Tenant("travel-b", domain="travel"))
    ca, cb = FlywheelClient(eng, "retail-a"), FlywheelClient(eng, "travel-b")
    for n in range(40):
        (ca if n % 2 else cb).report(input_text=f"prompt {n}",
                                     output_text=f"answer {n}", reward=0.9,
                                     cross_learning=f"lesson {n}")
    eng.flush()
    eng.pull("retail-a")   # generate lift ledger state
    eng.save(path)

    snap = {
        "threshold": eng.threshold.value,
        "promotions": eng.health()["promotions"],
        "lift_contrib": eng.lift.lift("retail-a")["contributed"],
        "reward_mean_a": eng.rewards.mean("retail-a"),
    }

    # restart: full learned state must survive, not just hub+accel
    eng2 = FlywheelEngine()
    eng2.load(path)
    assert eng2.threshold.value == snap["threshold"]
    assert eng2.health()["promotions"] == snap["promotions"]
    assert eng2.lift.lift("retail-a")["contributed"] == snap["lift_contrib"]
    assert eng2.rewards.mean("retail-a") == snap["reward_mean_a"]
