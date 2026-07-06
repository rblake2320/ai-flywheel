"""k-anonymity gate, untrusted-inbound marking, and sanitizer hardening."""
from aiflywheel.learning.hub import CrossLearningHub, SharedLearning
from aiflywheel.safety.sanitizer import LearningSanitizer


def _ln(tenant, lesson, domain="retail"):
    return SharedLearning(source_tenant=tenant, domain=domain, lesson=lesson, reward_score=0.9)


def test_k_anonymity_holds_until_k_similar():
    hub = CrossLearningHub(min_k=3)
    assert hub.contribute(_ln("a", "discount timing lifts conversion")) is False
    assert hub.contribute(_ln("b", "discount timing lifts conversion")) is False
    assert hub.coverage()["total_learnings"] == 0        # still below k
    assert hub.coverage()["pending_below_k"] == 2
    # third similar learning promotes the whole bucket
    assert hub.contribute(_ln("c", "discount timing lifts conversion")) is True
    assert hub.coverage()["total_learnings"] == 3


def test_k1_distributes_immediately():
    hub = CrossLearningHub(min_k=1)
    assert hub.contribute(_ln("a", "lesson")) is True
    assert hub.coverage()["total_learnings"] == 1


def test_inbound_marked_untrusted():
    hub = CrossLearningHub(min_k=1)
    hub.contribute(_ln("a", "lesson one"))
    got = hub.distribute("b")
    assert got and all(ln.trusted is False for ln in got)   # untrusted inbound


def test_distribute_excludes_own():
    hub = CrossLearningHub(min_k=1)
    hub.contribute(_ln("a", "from a"))
    hub.contribute(_ln("b", "from b"))
    got = hub.distribute("a")
    assert all(ln.source_tenant != "a" for ln in got)


def test_sanitizer_normalizes_zero_width_evasion():
    s = LearningSanitizer(hard_block=frozenset({"comp plan"}))
    # zero-width space inserted between the two words of a blocked phrase; after
    # NFKC + zero-width strip it collapses to "comp plan" and IS caught.
    r2 = s.sanitize("the comp​ plan value")
    assert r2.ok is False and r2.reason.startswith("hard_block")


def test_sanitizer_catches_ssn_and_card():
    s = LearningSanitizer()
    r = s.sanitize("pattern holds; ref 123-45-6789 and card 4111 1111 1111 1111 seen")
    assert "123-45-6789" not in r.text
    assert "4111 1111 1111 1111" not in r.text


def test_sanitizer_pluggable_detector_redacts_names():
    class FakeNER:
        def detect(self, text):
            return ["Priscilla Ashworth", "Meridian Realty"]

    s = LearningSanitizer(detector=FakeNER())
    r = s.sanitize("the deal with Priscilla Ashworth at Meridian Realty closed well")
    assert r.ok
    assert "Priscilla Ashworth" not in r.text
    assert "Meridian Realty" not in r.text
