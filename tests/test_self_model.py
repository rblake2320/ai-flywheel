"""v0.8.0: operational self-awareness — a grounded, self-checking self-model.

NOT sentience. Every property must be grounded in real measured state, and the
self-check must actually catch inconsistencies (a self-model that can't catch
itself being wrong is worthless).
"""
from aiflywheel.contract.sdk import FlywheelClient
from aiflywheel.core.learner import FewShotLearner, SimulatedLearner
from aiflywheel.engine import FlywheelEngine
from aiflywheel.introspection.self_model import SelfModel
from aiflywheel.metrics.promotion import PromotionGate
from aiflywheel.reflection.recall import RemoteRecall
from aiflywheel.reflection.whycase import WhyStore
from aiflywheel.tenancy.tenant import Tenant


def _engine(**kw):
    eng = FlywheelEngine(batch_size=10, learner=FewShotLearner(), **kw)
    return eng


def test_capabilities_reflect_actual_wiring():
    bare = _engine()
    caps = bare.self_model().capabilities()
    assert caps["self_correct"] is False          # no promotion gate
    assert caps["self_reflect"] is False          # no reflector
    assert caps["can_really_train"] is True        # FewShotLearner is real

    wired = _engine(promotion=PromotionGate(), reflector=WhyStore.__new__(WhyStore))
    # give the reflector a real store
    import tempfile
    wired.reflector = WhyStore(outbox_dir=tempfile.mkdtemp())
    caps2 = wired.self_model().capabilities()
    assert caps2["self_correct"] is True
    assert caps2["self_reflect"] is True


def test_simulated_learner_knows_it_cant_really_train():
    eng = FlywheelEngine(learner=SimulatedLearner())
    assert eng.self_model().capabilities()["can_really_train"] is False


def test_confidence_is_low_with_no_data():
    eng = _engine()
    c = eng.self_model().confidence()
    assert c["score"] == 0.0                        # nothing trained yet
    assert c["verdict"] == "unsure"


def test_confidence_rises_with_real_training():
    eng = _engine(promotion=PromotionGate())
    eng.add_tenant(Tenant("a", domain="retail"))
    eng.add_tenant(Tenant("b", domain="travel"))
    ca, cb = FlywheelClient(eng, "a"), FlywheelClient(eng, "b")
    for n in range(60):
        (ca if n % 2 else cb).report(input_text=f"p{n}", output_text=f"o{n}",
                                     reward=0.9, cross_learning=f"l{n}")
    eng.flush()
    c = eng.self_model().confidence()
    assert c["score"] > 0.0
    assert 0.0 <= c["score"] <= 1.0


def test_known_gaps_names_real_blind_spots():
    eng = _engine()                                 # nothing trained, 0 tenants
    gaps = eng.self_model().known_gaps()
    joined = " ".join(gaps)
    assert "no training" in joined
    assert "not networked" in joined
    assert "promotion gate" in joined               # none wired


def test_self_check_catches_unrecorded_rollback():
    # rollbacks without WhyCases = a real inconsistency the model must catch
    class DegradingLearner:
        def __init__(self):
            self._q = 0.8
        def quality(self):
            return self._q
        def train(self, b):
            self._q = 0.3
            return self._q
        def snapshot(self):
            return self._q
        def rollback(self, s):
            self._q = s

    # promotion gate but NO reflector → rollback happens, no WhyCase recorded
    eng = FlywheelEngine(batch_size=5, learner=DegradingLearner(),
                         promotion=PromotionGate())
    eng.add_tenant(Tenant("t", domain="retail"))
    c = FlywheelClient(eng, "t")
    for n in range(5):
        c.report(input_text=f"q{n}", output_text=f"a{n}", reward=0.9, domain="retail")
    # 1 rollback occurred; with no reflector, why_cases stays 0 — but the model
    # only flags the mismatch when a reflector IS present, so wire one late:
    import tempfile
    eng.reflector = WhyStore(outbox_dir=tempfile.mkdtemp())
    check = eng.self_model().self_check()
    assert check["consistent"] is False
    assert any("unrecorded" in c for c in check["contradictions"])


def test_self_check_passes_on_coherent_engine():
    eng = _engine(promotion=PromotionGate())
    eng.add_tenant(Tenant("a", domain="retail"))
    c = FlywheelClient(eng, "a")
    for n in range(10):
        c.report(input_text=f"p{n}", output_text=f"o{n}", reward=0.9, domain="retail")
    eng.flush()
    assert eng.self_model().self_check()["consistent"] is True


def test_confidence_drops_when_recall_blind():
    from aiflywheel.core.interaction import Interaction

    eng = _engine(promotion=PromotionGate(),
                  recall=RemoteRecall(base_url="http://127.0.0.1:59999", timeout=0.1))
    eng.add_tenant(Tenant("a", domain="retail"))
    c = FlywheelClient(eng, "a")
    for n in range(10):
        c.report(input_text=f"p{n}", output_text=f"o{n}", reward=0.9, domain="retail")
    eng.flush()
    # a recall against the down brain forces a fallback the model can feel
    eng.seen_regression_before([Interaction(id="x", tenant_id="a", timestamp="",
                                            domain="retail")])
    model = eng.self_model()
    assert model.state()["recall_fallbacks"] >= 1
    assert model.confidence()["basis"]["blind_penalty"] < 1.0


def test_full_report_is_honest_about_identity():
    rep = SelfModel(_engine()).report()
    assert "not sentient" in rep["identity"]
    assert "capabilities" in rep and "confidence" in rep and "self_check" in rep
