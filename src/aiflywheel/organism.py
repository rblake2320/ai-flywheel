"""
The Organism — everything connects to everything (the Disney model, in code).

Until now the faculties were built and wired, but a run was mostly a straight
pipeline. This makes it a CONNECTED CYCLE where each faculty feeds the others:

    ingest ─▶ learn(exploit) ─▶ introspect(self-model) ─┐
      ▲                                                  │ confidence + gaps
      │ experiments                                      ▼
      └──────────── explore(seek novelty) ◀── frontiers ─┘
                         │
        reflect(WhyCases)┘── feed the curator, avoid past mistakes

The connections that make it an organism, not a stack:
  - self-model GAPS drive what the explorer investigates (introspection → curiosity)
  - hub LEARNINGS become the explorer's coverage (experience → what's worth exploring)
  - CONFIDENCE governs the explore/exploit balance — low confidence explores more,
    high confidence exploits (self-awareness → growth mode); this is how self-LEARN
    (targeted) and self-EXPLORE (open-ended) share one axis
  - explorer FRONTIERS become next-cycle experiments (curiosity → new experience)
  - WhyCases feed the curator (reflection → curation)

Emergent safety, for free: explored hypotheses are SYNTHETIC experience
(provenance="synthetic"), so the RealDataFloor automatically keeps exploration
from dominating or collapsing the model. Curiosity is bounded by the same guard
that prevents model collapse — nobody had to add a separate limiter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aiflywheel.core.interaction import Interaction
from aiflywheel.explore.curiosity import Explorer, Hypothesis


@dataclass
class CycleReport:
    ingested: int
    trained_batches: int
    confidence: float
    verdict: str
    gaps: list[str]
    self_consistent: bool
    explore_budget: int
    frontiers: list[Hypothesis]
    connections_fired: list[str] = field(default_factory=list)


@dataclass
class Organism:
    """Runs the whole flywheel as one connected, self-directing cycle."""

    engine: object
    explorer: Explorer = field(default_factory=Explorer)

    def run_cycle(self, events: list[Interaction] | None = None) -> CycleReport:
        conns: list[str] = []
        e = self.engine

        # 1) INGEST — real experience in
        ingested = 0
        for it in (events or []):
            try:
                e.submit(it)
                ingested += 1
            except KeyError:
                continue
        if ingested:
            conns.append(f"ingest→learn: {ingested} experiences")

        # 2) LEARN (exploit) — train on what we have
        before = e.accel.report()["batches"]
        e.flush()
        trained = e.accel.report()["batches"] - before
        if trained:
            conns.append(f"learn→introspect: {trained} batch(es) trained")

        # 3) INTROSPECT — the self-model reads the wheel's own state
        sm = e.self_model()
        conf = sm.confidence()
        gaps = sm.known_gaps()
        check = sm.self_check()
        conns.append(f"introspect→explore: confidence {conf['score']} governs explore/exploit")

        # 4) EXPLORE (seek novelty) — coverage from experience, frontiers from gaps
        self.explorer.learn_coverage(e)
        conns.append("experience→coverage: hub learnings mapped")

        # 5) BALANCE — confidence sets how much to explore vs exploit.
        #    low confidence -> explore MORE (we don't know enough yet).
        explore_budget = 1 + int(round((1.0 - conf["score"]) * 5))
        frontiers = self.explorer.frontiers(e, gaps=gaps, k=explore_budget)
        if frontiers:
            conns.append(f"gaps+coverage→frontiers: {len(frontiers)} novel directions")
        if any(f.source == "gap" for f in frontiers):
            conns.append("self-model gaps became exploration targets")

        return CycleReport(
            ingested=ingested,
            trained_batches=trained,
            confidence=conf["score"],
            verdict=conf["verdict"],
            gaps=gaps,
            self_consistent=check["consistent"],
            explore_budget=explore_budget,
            frontiers=frontiers,
            connections_fired=conns,
        )

    def frontiers_as_experiments(self, frontiers: list[Hypothesis]) -> list[Interaction]:
        """Turn chosen frontiers into SYNTHETIC experiments for the next cycle.

        Marked provenance='synthetic' on purpose: the RealDataFloor then bounds
        how much exploration can influence training — curiosity that can't cause
        collapse. Only frontiers with a concrete target domain become experiments.
        """
        out = []
        for i, f in enumerate(frontiers):
            if f.target_domain == "_meta":
                continue
            out.append(Interaction(
                id=f"explore-{i}", tenant_id="_explorer", timestamp="",
                input_text=f.description, output_text=f"[hypothesis:{f.source}]",
                reward_score=f.novelty, domain=f.target_domain,
                cross_learning=f"exploration: {f.rationale}",
                provenance="synthetic",
            ))
        return out
