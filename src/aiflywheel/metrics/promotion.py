"""
PromotionGate — closes the loop: promote a better model, roll back a worse one.

Model-collapse defense (RealDataFloor) is PREVENTIVE — it stops obviously-bad
batches from training. The PromotionGate is the CORRECTIVE backstop: after a
train, it decides whether the new model is actually an improvement and, if not,
rolls the learner back to the pre-train snapshot. This is what makes the wheel a
self-improving system that *cannot make itself worse* — you can train freely
because every train is provisional until it proves itself.

Decision, in order of strength:
  1. If a Judge + golden set + an answerable model are available → run the
     RegressionGate (must not backslide on frozen cases). A regression = ROLLBACK.
  2. Otherwise fall back to a quality-delta rule: ROLLBACK if quality dropped
     more than `tolerance`, else PROMOTE.
"""
from __future__ import annotations

from dataclasses import dataclass

from aiflywheel.metrics.judge import Judge, RegressionGate

PROMOTE = "PROMOTE"
ROLLBACK = "ROLLBACK"


@dataclass
class PromotionDecision:
    action: str
    reason: str
    prev_quality: float
    new_quality: float
    regression: dict | None = None


@dataclass
class PromotionGate:
    """Decides PROMOTE vs ROLLBACK after a training step."""

    tolerance: float = 0.0        # allowed quality dip before rollback (delta rule)
    judge: Judge | None = None
    golden: RegressionGate | None = None

    def decide(
        self,
        prev_quality: float,
        new_quality: float,
        answer_fn=None,
    ) -> PromotionDecision:
        # strongest signal: frozen golden-set regression check
        if self.judge is not None and self.golden is not None and answer_fn is not None:
            result = self.golden.evaluate(self.judge, answer_fn)
            if not result["passed"]:
                return PromotionDecision(
                    ROLLBACK, "golden-set regression", prev_quality, new_quality, result
                )
            # passed the gate AND didn't lose quality → promote
            if new_quality + self.tolerance < prev_quality:
                return PromotionDecision(
                    ROLLBACK, "quality dropped despite passing golden set",
                    prev_quality, new_quality, result,
                )
            return PromotionDecision(PROMOTE, "passed golden set", prev_quality,
                                     new_quality, result)

        # fallback: quality-delta rule
        if new_quality + self.tolerance < prev_quality:
            return PromotionDecision(ROLLBACK, "quality regressed", prev_quality, new_quality)
        return PromotionDecision(PROMOTE, "quality held or improved", prev_quality, new_quality)
