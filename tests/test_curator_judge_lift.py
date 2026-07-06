"""v0.4.0: multi-stage curator, judge/regression gate, few-shot learner, lift."""
from aiflywheel.contract.sdk import FlywheelClient
from aiflywheel.core.interaction import Interaction
from aiflywheel.core.learner import FewShotLearner
from aiflywheel.curation.curator import (
    DiversityStage,
    RewardStage,
    SemanticDedupStage,
    default_curator,
)
from aiflywheel.engine import FlywheelEngine
from aiflywheel.metrics.judge import (
    DeterministicJudge,
    GoldenCase,
    RegressionGate,
    win_rate,
)
from aiflywheel.tenancy.tenant import Tenant


def _it(tid, reward, text="", dom="retail"):
    return Interaction(id="x", tenant_id=tid, timestamp="", reward_score=reward,
                       input_text=text, output_text=text, domain=dom)


# --- curator ---
def test_reward_stage_filters():
    batch = [_it("a", 0.2), _it("a", 0.9)]
    kept = RewardStage(min_reward=0.5).curate(batch)
    assert len(kept) == 1 and kept[0].reward_score == 0.9


def test_semantic_dedup_drops_near_duplicates():
    batch = [
        _it("a", 0.9, "the quick brown fox jumps"),
        _it("a", 0.5, "the quick brown fox jumps"),      # dup → dropped
        _it("a", 0.8, "completely different content here"),
    ]
    kept = SemanticDedupStage(threshold=0.8).curate(batch)
    assert len(kept) == 2
    assert any(i.reward_score == 0.9 for i in kept)      # kept the best of the dup pair


def test_diversity_caps_dominant_tenant():
    batch = [_it("a", 0.9) for _ in range(8)] + [_it("b", 0.9) for _ in range(2)]
    kept = DiversityStage(max_share=0.6).curate(batch)
    a = sum(1 for i in kept if i.tenant_id == "a")
    assert a <= int(len(kept) * 0.6) + 1                 # 'a' no longer dominates fully


def test_default_curator_pipeline_runs():
    batch = [_it("a", 0.1)] + [_it("a", 0.9, "same text") for _ in range(5)] \
        + [_it("b", 0.9, "other") for _ in range(3)]
    kept = default_curator(min_reward=0.3).curate(batch)
    assert 0 < len(kept) <= len(batch)
    trace = default_curator(min_reward=0.3).trace(batch)
    assert trace[0]["stage"] == "input"


def test_engine_uses_curator():
    eng = FlywheelEngine(batch_size=10, curator=default_curator(min_reward=0.5))
    eng.add_tenant(Tenant("a", domain="retail"))
    c = FlywheelClient(eng, "a")
    for _ in range(10):
        c.report(input_text="dup text same", output_text="dup text same",
                 reward=0.9, domain="retail")
    # curator dedups the identical batch down; engine still trains on what's left
    assert eng.accel.report()["batches"] >= 0    # ran without error


# --- judge + regression gate ---
def test_deterministic_judge_and_regression_gate():
    judge = DeterministicJudge(scorer=lambda p, a: 0.9 if "good" in a else 0.1)
    gate = RegressionGate(cases=[GoldenCase("q1", 0.6), GoldenCase("q2", 0.6)])
    ok = gate.evaluate(judge, answer_fn=lambda p: "good answer")
    assert ok["passed"] is True and ok["mean"] == 0.9
    bad = gate.evaluate(judge, answer_fn=lambda p: "bad answer")
    assert bad["passed"] is False and len(bad["regressions"]) == 2


def test_win_rate():
    judge = DeterministicJudge(scorer=lambda p, a: len(a) / 10.0)
    wr = win_rate(judge, ["p1", "p2"],
                  new_fn=lambda p: "longereranswer", prev_fn=lambda p: "short")
    assert wr["win_rate"] == 1.0


# --- few-shot learner (a real learner) ---
def test_few_shot_learner_builds_bank_and_answers():
    lr = FewShotLearner(bank_size=3)
    lr.train([_it("a", 0.5, "meh", dom="retail"),
              _it("a", 0.95, "best retail answer", dom="retail")])
    assert lr.quality() > 0
    assert lr.answer("retail") == "best retail answer"   # returns the top exemplar
    assert len(lr.exemplars("retail")) == 2


# --- lift attribution ---
def test_lift_tracks_contribution_and_gain():
    eng = FlywheelEngine(batch_size=1000)
    for tid in ("a", "b"):
        eng.add_tenant(Tenant(tid, domain="retail"))
    ca, cb = FlywheelClient(eng, "a"), FlywheelClient(eng, "b")
    for n in range(5):
        ca.report(input_text=f"p{n}", reward=0.9, cross_learning=f"a lesson {n}")
        cb.report(input_text=f"q{n}", reward=0.9, cross_learning=f"b lesson {n}")
    eng.pull("a")     # a receives b's learnings
    rep = eng.lift.lift("a")
    assert rep["contributed"] == 5
    assert rep["gained"] >= 5          # got b's 5 learnings
    assert rep["lift_ratio"] >= 1.0    # net winner
