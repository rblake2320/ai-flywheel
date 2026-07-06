"""TRLLearner conforms to the Learner protocol; heavy training is skipped if TRL absent."""
import importlib.util

import pytest

from aiflywheel.backends.trl_learner import TRLLearner
from aiflywheel.core.interaction import Interaction
from aiflywheel.core.learner import Learner

_HAS_TRL = importlib.util.find_spec("trl") is not None


def _it(prompt, completion, reward=0.9):
    return Interaction(id="x", tenant_id="t", timestamp="", input_text=prompt,
                       output_text=completion, reward_score=reward, domain="retail")


def test_conforms_to_learner_protocol():
    lr = TRLLearner()
    assert isinstance(lr, Learner)        # structural: has train() + quality()
    assert lr.quality() == 0.0            # untrained


def test_row_conversion_filters_correctly():
    lr = TRLLearner(min_reward=0.5)
    rows = lr._to_rows([
        _it("q1", "a1", reward=0.9),      # kept
        _it("q2", "a2", reward=0.2),      # below min_reward → dropped
        _it("q3", "", reward=0.9),        # empty completion → dropped
    ])
    assert len(rows) == 1
    assert "q1" in rows[0]["text"] and "a1" in rows[0]["text"]


def test_empty_batch_is_noop():
    lr = TRLLearner()
    assert lr.train([]) == 0.0


@pytest.mark.skipif(not _HAS_TRL, reason="trl not installed (install ai-flywheel[trainer])")
def test_real_smoke_train():
    lr = TRLLearner(base_model="HuggingFaceTB/SmolLM2-135M", epochs=1.0)
    batch = [_it(f"question {i}", f"answer {i}") for i in range(4)]
    q = lr.train(batch)
    assert 0.0 <= q <= 1.0
    assert lr.quality() == q
