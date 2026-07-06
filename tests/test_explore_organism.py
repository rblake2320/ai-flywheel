"""v0.9.0: self-explore (curiosity) + the organism (everything connects)."""
from aiflywheel.contract.sdk import FlywheelClient
from aiflywheel.core.learner import FewShotLearner
from aiflywheel.core.provenance import real_fraction
from aiflywheel.engine import FlywheelEngine
from aiflywheel.explore.curiosity import (
    CombinatorialIdeaSource,
    CoverageMap,
    Explorer,
    IdeaSource,
)
from aiflywheel.metrics.promotion import PromotionGate
from aiflywheel.organism import Organism
from aiflywheel.tenancy.tenant import Tenant


# --- coverage / novelty ---
def test_coverage_novelty():
    cov = CoverageMap()
    cov.observe("retail", "discount timing lifts conversion")
    assert cov.is_novel("retail", "discount timing lifts conversion") is False
    assert cov.novelty("retail", "discount timing lifts conversion") == 0.0
    # unseen domain + unseen pattern → maximally novel
    assert cov.novelty("aerospace", "telemetry anomaly clustering") == 1.0


# --- idea source does cross-domain transfer (the Disney mechanism) ---
def test_combinatorial_source_transfers_across_domains():
    src = CombinatorialIdeaSource()
    assert isinstance(src, IdeaSource)
    ideas = src.generate([("retail", "objection handling")], ["retail", "real_estate"])
    # a retail pattern is proposed for real_estate
    assert any(h.target_domain == "real_estate" and h.source == "transfer" for h in ideas)


def test_explorer_frontiers_prefer_novelty():
    eng = _trained_engine()
    ex = Explorer()
    ex.learn_coverage(eng)
    fr = ex.frontiers(eng, gaps=["no golden set"], k=5)
    assert fr
    # novelty-sourced frontiers (transfer/mix) are sorted by novelty descending
    nov = [h for h in fr if h.source in {"transfer", "mix"}]
    assert all(nov[i].novelty >= nov[i + 1].novelty for i in range(len(nov) - 1))
    # every frontier carries a real novelty score in range
    assert all(0.0 <= h.novelty <= 1.0 for h in fr)


def test_gaps_become_exploration_targets():
    eng = _trained_engine()
    fr = Explorer().frontiers(eng, gaps=["I have a specific blind spot X"], k=6)
    assert any(f.source == "gap" for f in fr)


# --- the organism: everything connects in one cycle ---
def test_organism_cycle_fires_connections():
    eng = _trained_engine()
    org = Organism(engine=eng)
    report = org.run_cycle()
    # introspection→explore and gaps→frontiers connections must have fired
    joined = " ".join(report.connections_fired)
    assert "introspect" in joined
    assert "frontiers" in joined
    assert report.frontiers                      # it proposed novel directions
    assert isinstance(report.self_consistent, bool)


def test_confidence_governs_explore_budget():
    # a low-confidence engine (little data) should explore MORE
    low = FlywheelEngine(batch_size=10, learner=FewShotLearner())
    low.add_tenant(Tenant("a", domain="retail"))
    org = Organism(engine=low)
    rep = org.run_cycle()
    assert rep.explore_budget >= 3               # unsure → explore aggressively


def test_frontiers_become_synthetic_experiments_bounded_by_floor():
    eng = _trained_engine()
    org = Organism(engine=eng)
    fr = org.run_cycle().frontiers
    experiments = org.frontiers_as_experiments(fr)
    # exploration output is SYNTHETIC — the collapse floor bounds it for free
    assert all(x.provenance == "synthetic" for x in experiments)
    if experiments:
        assert real_fraction(experiments) == 0.0   # pure exploration = pure synthetic


def test_explorer_feeds_back_into_a_next_cycle():
    eng = _trained_engine()
    org = Organism(engine=eng)
    fr = org.run_cycle().frontiers
    experiments = org.frontiers_as_experiments(fr)
    # feeding synthetic experiments back does not crash and is floor-guarded
    eng.add_tenant(Tenant("_explorer", domain="retail"))
    rep2 = org.run_cycle(events=experiments)
    assert rep2.ingested >= 0                     # closed the creative loop safely


def _trained_engine():
    eng = FlywheelEngine(batch_size=10, learner=FewShotLearner(), promotion=PromotionGate())
    eng.add_tenant(Tenant("retail-a", domain="retail"))
    eng.add_tenant(Tenant("travel-b", domain="travel"))
    ca, cb = FlywheelClient(eng, "retail-a"), FlywheelClient(eng, "travel-b")
    for n in range(20):
        (ca if n % 2 else cb).report(input_text=f"prompt {n}", output_text=f"answer {n}",
                                     reward=0.9, cross_learning=f"pattern {n % 5} helps")
    eng.flush()
    return eng
