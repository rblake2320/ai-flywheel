"""The self-tuning valve holds a target accept rate and stays bounded."""
from aiflywheel.adaptive.threshold import AdaptiveThreshold


def test_tightens_when_too_much_passes():
    t = AdaptiveThreshold(value=0.5, target_accept_rate=0.3)
    for _ in range(100):        # everything passes at 0.5 -> accept rate 1.0
        t.observe(0.9)
    new = t.update()
    assert new > 0.5           # raised the bar


def test_loosens_when_too_little_passes():
    t = AdaptiveThreshold(value=0.9, target_accept_rate=0.3)
    for _ in range(100):        # nothing passes at 0.9 -> accept rate 0.0
        t.observe(0.4)
    new = t.update()
    assert new < 0.9           # lowered the bar


def test_stays_within_bounds():
    t = AdaptiveThreshold(value=0.94, upper_bound=0.95, step=0.5)
    for _ in range(50):
        t.observe(1.0)
    assert t.update() <= 0.95
    t2 = AdaptiveThreshold(value=0.51, lower_bound=0.5, step=0.5)
    for _ in range(50):
        t2.observe(0.0)
    assert t2.update() >= 0.5


def test_converges_toward_target():
    t = AdaptiveThreshold(value=0.5, target_accept_rate=0.3, step=0.05)
    # scores uniformly spread 0..1; ~top 30% should end up passing
    for _ in range(60):
        for i in range(10):
            t.observe(i / 10.0)
        t.update()
    assert 0.6 <= t.value <= 0.8
