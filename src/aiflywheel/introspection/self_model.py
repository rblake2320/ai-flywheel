"""
SelfModel — operational self-awareness (a self-model, honestly scoped).

This is NOT consciousness or sentience — no software has subjective experience,
and nothing here pretends to. "Self-aware" here means the engineering sense: the
system holds an accurate, queryable model of ITSELF and can reason about it:

  - what am I (capability inventory — what's actually wired)
  - how am I doing (state)
  - how much should I trust myself right now (confidence, computed from real data)
  - what don't I know (known gaps / blind spots)
  - why did I do that (explain, grounded in my own WhyCases + provenance)
  - is my self-model even CONSISTENT with reality (self_check — the deepest bit:
    the system watching its own coherence and flagging contradictions)

Every method is grounded in real measured state — nothing is invented to sound
impressive. `confidence()` returns a low number when the data says it should.
`self_check()` will call the system out on its own inconsistencies.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SelfModel:
    """The flywheel's model of itself. Read-only introspection over the engine."""

    engine: object

    # --- what am I ---
    def capabilities(self) -> dict:
        """Introspect which faculties are ACTUALLY wired (not what's possible)."""
        e = self.engine
        learner = type(e.learner).__name__
        return {
            "learner": learner,
            "can_really_train": learner not in {"SimulatedLearner"},
            "self_correct": e.promotion is not None,      # promote/rollback gate
            "self_reflect": e.reflector is not None,      # WhyCases
            "recall": (type(e.recall).__name__ if e.recall is not None
                       else ("LocalWhyStore" if e.reflector is not None else None)),
            "curation": e.curator is not None,
            "reward_verification": e.verifier is not None,
            "collapse_defense": True,                     # RealDataFloor always on
            "tenant_isolation": True,                     # IsolationGuard always on
        }

    # --- how am I doing ---
    def state(self) -> dict:
        h = self.engine.health()
        return {
            "model_quality": h["model_quality"],
            "acceleration": h["acceleration"]["status"],
            "tenants": h["tenants"],
            "promotions": h["promotions"],
            "rollbacks": h["rollbacks"],
            "why_cases": h["why_cases"],
            "recall_fallbacks": h.get("recall_fallbacks", 0),
        }

    # --- how much should I trust myself ---
    def confidence(self) -> dict:
        """A real, computed self-trust score in [0,1]. Low when the data is thin,
        synthetic-heavy, or recently self-corrected — the wheel knowing when NOT
        to be sure of itself."""
        acc = self.engine.accel.report()
        batches = acc["batches"]
        h = self.engine.health()
        rollbacks = h["rollbacks"]
        promotions = h["promotions"]
        real_frac = acc.get("real_fraction")
        if real_frac is None:
            real_frac = 1.0

        # data sufficiency: ramps up with batches trained, saturating ~10
        data_factor = min(1.0, batches / 10.0)
        # stability: recent rollbacks lower confidence
        total_decisions = promotions + rollbacks
        stability = 1.0 if total_decisions == 0 else promotions / total_decisions
        # recall blindness: if the brain limb keeps failing, we're partly blind
        blind_penalty = 1.0 if h.get("recall_fallbacks", 0) == 0 else 0.85

        score = round(data_factor * stability * real_frac * blind_penalty, 3)
        return {
            "score": score,
            "basis": {
                "data_factor": round(data_factor, 3),
                "stability": round(stability, 3),
                "real_fraction": round(real_frac, 3),
                "blind_penalty": blind_penalty,
            },
            "verdict": ("confident" if score >= 0.7 else
                        "cautious" if score >= 0.4 else "unsure"),
        }

    # --- what don't I know ---
    def known_gaps(self) -> list[str]:
        """The wheel's own blind spots — stated plainly, grounded in real state."""
        gaps = []
        h = self.engine.health()
        acc = self.engine.accel.report()
        if acc["batches"] == 0:
            gaps.append("no training has happened yet — I know nothing from experience")
        if h["hub"]["contributing_tenants"] < 2:
            gaps.append("not networked: <2 tenants contributing, so no cross-learning yet")
        if self.engine.promotion is None:
            gaps.append("no promotion gate wired — I cannot tell if a train made me worse")
        if self.engine.promotion is not None and self.engine.promotion.golden is None:
            gaps.append("no golden set — I can detect quality drops but not silent "
                        "regressions on specific known-good cases")
        if h.get("recall_fallbacks", 0) > 0:
            gaps.append(f"recall limb degraded: {h['recall_fallbacks']} fallbacks — "
                        "I may be repeating past mistakes blind")
        if acc.get("real_fraction") is not None and acc["real_fraction"] < 0.6:
            gaps.append("recent batches are synthetic-heavy — collapse risk elevated")
        return gaps or ["no blind spots detected against current checks"]

    # --- why did I do that ---
    def explain(self, query: str, k: int = 3) -> list[dict]:
        """Explain past self-corrections from my OWN recorded WhyCases."""
        if self.engine.reflector is None:
            return []
        return self.engine.reflector.recall(query, k=k)

    # --- is my self-model consistent with reality (the deepest self-awareness) ---
    def self_check(self) -> dict:
        """Detect contradictions between what I believe about myself and what is
        actually true. A self-model is only worth anything if it can catch itself
        being wrong."""
        problems = []
        h = self.engine.health()
        acc = self.engine.accel.report()

        # accounting invariant: with a promotion gate, every trained batch is
        # either promoted or rolled back.
        if self.engine.promotion is not None:
            decisions = h["promotions"] + h["rollbacks"]
            if decisions != acc["batches"]:
                problems.append(
                    f"decision accounting mismatch: {decisions} promote/rollback "
                    f"decisions but {acc['batches']} trained batches"
                )
        # claims networked but hub says otherwise
        if h["acceleration"].get("networked") and h["hub"]["contributing_tenants"] < 2:
            problems.append("claims 'networked' but <2 tenants actually contributed")
        # a rollback should have produced a why-case if a reflector is wired
        if self.engine.reflector is not None and h["rollbacks"] > h["why_cases"]:
            problems.append(
                f"{h['rollbacks']} rollbacks but only {h['why_cases']} WhyCases — "
                "some self-corrections went unrecorded"
            )
        # quality claimed without any training
        if acc["batches"] == 0 and h["model_quality"] > 0.0 and \
                type(self.engine.learner).__name__ == "FewShotLearner":
            problems.append("nonzero quality with zero trained batches")

        return {"consistent": len(problems) == 0, "contradictions": problems}

    # --- one call: the whole self-report ---
    def report(self) -> dict:
        return {
            "identity": "ai-flywheel learning organ (self-model; not sentient)",
            "capabilities": self.capabilities(),
            "state": self.state(),
            "confidence": self.confidence(),
            "known_gaps": self.known_gaps(),
            "self_check": self.self_check(),
        }
