"""v0.5.0: the loop closes — promote a better model, roll back a worse one."""
from aiflywheel.contract.sdk import FlywheelClient
from aiflywheel.core.interaction import Interaction
from aiflywheel.core.learner import FewShotLearner, Revertable, SimulatedLearner
from aiflywheel.core.reward import JudgeRewardVerifier
from aiflywheel.engine import FlywheelEngine
from aiflywheel.metrics.judge import DeterministicJudge, GoldenCase, RegressionGate
from aiflywheel.metrics.promotion import PROMOTE, ROLLBACK, PromotionGate
from aiflywheel.tenancy.tenant import Tenant


# --- learners are revertable ---
def test_learners_snapshot_and_rollback():
    for lr in (SimulatedLearner(start=0.5), FewShotLearner()):
        assert isinstance(lr, Revertable)
        snap = lr.snapshot()
        lr.train([Interaction(id="x", tenant_id="t", timestamp="",
                              input_text="p", output_text="o", reward_score=0.9,
                              domain="retail")])
        lr.rollback(snap)
        assert lr.quality() == (0.5 if isinstance(lr, SimulatedLearner) else 0.0)


# --- promotion gate decisions ---
def test_gate_promotes_on_improvement():
    d = PromotionGate().decide(prev_quality=0.5, new_quality=0.7)
    assert d.action == PROMOTE


def test_gate_rolls_back_on_regression():
    d = PromotionGate(tolerance=0.0).decide(prev_quality=0.7, new_quality=0.5)
    assert d.action == ROLLBACK
    assert "regress" in d.reason


def test_gate_rolls_back_on_golden_regression():
    judge = DeterministicJudge(scorer=lambda p, a: 0.1)   # everything fails
    gate = PromotionGate(judge=judge,
                         golden=RegressionGate(cases=[GoldenCase("q", 0.6)]))
    d = gate.decide(0.5, 0.9, answer_fn=lambda p: "answer")
    assert d.action == ROLLBACK
    assert "golden" in d.reason


# --- engine rolls back a regressing batch (cannot make itself worse) ---
def test_engine_rolls_back_regression():
    # a learner whose quality DROPS when trained (simulates a bad batch)
    class DegradingLearner:
        def __init__(self):
            self._q = 0.8
        def quality(self):
            return self._q
        def train(self, batch):
            self._q = 0.3          # training made it worse
            return self._q
        def snapshot(self):
            return self._q
        def rollback(self, snap):
            self._q = snap

    eng = FlywheelEngine(batch_size=5, learner=DegradingLearner(),
                         promotion=PromotionGate(tolerance=0.0))
    eng.add_tenant(Tenant("t", domain="retail"))
    c = FlywheelClient(eng, "t")
    for _ in range(5):
        c.report(input_text="p", output_text="o", reward=0.9, domain="retail")
    h = eng.health()
    assert h["rollbacks"] == 1
    assert h["model_quality"] == 0.8      # rolled back to pre-train quality


def test_engine_promotes_improvement():
    eng = FlywheelEngine(batch_size=10, learner=SimulatedLearner(start=0.5),
                         promotion=PromotionGate())
    eng.add_tenant(Tenant("t", domain="retail"))
    c = FlywheelClient(eng, "t")
    for n in range(10):
        c.report(input_text=f"p{n}", output_text=f"o{n}", reward=0.9, domain="retail")
    h = eng.health()
    assert h["promotions"] >= 1
    assert h["rollbacks"] == 0


# --- real reward source: judge scores instead of trusting the tenant ---
def test_judge_reward_verifier_overrides_inflated_reward():
    # judge says the answer is bad (0.1); tenant claims 1.0
    judge = DeterministicJudge(scorer=lambda p, a: 0.1)
    v = JudgeRewardVerifier(judge=judge, weight=1.0, prompt="q", answer="bad")
    assert v.verify("t", reported=1.0) == 0.1     # judge wins, inflation blocked


def test_judge_reward_blends_with_tenant():
    judge = DeterministicJudge(scorer=lambda p, a: 0.5)
    v = JudgeRewardVerifier(judge=judge, weight=0.5, prompt="q", answer="a")
    assert v.verify("t", reported=1.0) == 0.75    # 0.5*0.5 + 0.5*1.0
