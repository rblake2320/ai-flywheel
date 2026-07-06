"""
Self-explore — open-ended, curiosity-driven exploration.

Self-LEARN is targeted research: improve toward a goal on the data you're given
(exploit). Self-EXPLORE is its opposite pole: with NO fixed target, seek what
hasn't been tried — the novel, the under-explored, the surprising. They are two
modes of one growth axis, balanced by the explore/exploit trade-off (see
organism.py; the self-model's confidence is the governor — low confidence pushes
toward explore, high toward exploit).

Grounded in real prior art, not hand-waving:
  - Novelty search / Quality-Diversity (MAP-Elites): keep an archive of what's
    been seen; value a candidate by how DIFFERENT it is from the nearest neighbor.
  - Intrinsic curiosity (Schmidhuber): prefer the learnable-but-not-yet-known.
  - Cross-domain recombination (the Disney mechanism): the richest novelty comes
    from transferring a pattern that worked in domain A into domain B — everything
    connecting to everything is literally how new hypotheses are generated.

The "invent something never done" part is a pluggable `IdeaSource`: the core ships
a deterministic combinatorial generator (real, no deps); an LLM/web-backed source
can be plugged behind the same protocol to inject genuinely open-ended ideas.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Hypothesis:
    """A proposed experiment to explore. Novelty-scored, with a rationale."""

    description: str
    target_domain: str
    source: str                 # where the idea came from (transfer/gap/mix/invent)
    novelty: float = 0.0        # 0..1, higher = more unexplored
    rationale: str = ""


class CoverageMap:
    """What the wheel has already explored — the archive novelty is measured against."""

    def __init__(self) -> None:
        self.domains: set[str] = set()
        self.patterns: set[str] = set()          # normalized pattern signatures
        self.domain_patterns: set[tuple[str, str]] = set()

    def observe(self, domain: str, pattern_text: str) -> None:
        sig = self._sig(pattern_text)
        self.domains.add(domain)
        self.patterns.add(sig)
        self.domain_patterns.add((domain, sig))

    @staticmethod
    def _sig(text: str) -> str:
        return " ".join(sorted(set(_WORD.findall((text or "").lower())))[:8])

    def is_novel(self, domain: str, pattern_text: str) -> bool:
        return (domain, self._sig(pattern_text)) not in self.domain_patterns

    def novelty(self, domain: str, pattern_text: str) -> float:
        """1.0 if the (domain, pattern) pair is wholly unseen; lower if familiar."""
        sig = self._sig(pattern_text)
        if (domain, sig) in self.domain_patterns:
            return 0.0
        # unseen pair; more novel if the pattern OR the domain is itself new
        n = 0.5
        if sig not in self.patterns:
            n += 0.25
        if domain not in self.domains:
            n += 0.25
        return round(n, 3)


@runtime_checkable
class IdeaSource(Protocol):
    """Generates candidate hypotheses. Core = combinatorial; plug an LLM/web one."""

    def generate(
        self, known_patterns: list[tuple[str, str]], domains: list[str]
    ) -> list[Hypothesis]:
        ...


@dataclass
class CombinatorialIdeaSource:
    """Deterministic idea generation: cross-domain transfer + pattern mixing.

    No dependencies, no randomness. This is the Disney mechanism made concrete —
    it proposes carrying each known pattern into every OTHER domain, and mixing
    pairs of patterns into combined hypotheses. A real LLM/web IdeaSource plugs in
    behind the same protocol for genuinely open-ended invention.
    """

    max_ideas: int = 25

    def generate(self, known_patterns, domains):
        ideas: list[Hypothesis] = []
        # 1) cross-domain transfer (everything connects to everything)
        for src_domain, pattern in known_patterns:
            for tgt in domains:
                if tgt == src_domain:
                    continue
                ideas.append(Hypothesis(
                    description=f"transfer '{pattern}' from {src_domain} into {tgt}",
                    target_domain=tgt, source="transfer",
                    rationale=f"a pattern that worked in {src_domain} may generalize to {tgt}",
                ))
        # 2) pattern mixing (recombine two known patterns into a novel one)
        for i in range(len(known_patterns)):
            for j in range(i + 1, len(known_patterns)):
                (d1, p1), (d2, p2) = known_patterns[i], known_patterns[j]
                ideas.append(Hypothesis(
                    description=f"combine '{p1}' + '{p2}'",
                    target_domain=d1, source="mix",
                    rationale=f"novel combination of a {d1} and a {d2} pattern",
                ))
        return ideas[: self.max_ideas]


@dataclass
class Explorer:
    """Seeks novel directions the wheel hasn't tried. The self-EXPLORE pole."""

    coverage: CoverageMap = field(default_factory=CoverageMap)
    idea_source: IdeaSource = field(default_factory=CombinatorialIdeaSource)

    def learn_coverage(self, engine) -> None:
        """Absorb what the wheel has already explored, from the hub's learnings."""
        for ln in engine.hub._learnings:
            self.coverage.observe(ln.domain, ln.lesson)

    def frontiers(self, engine, gaps: list[str] | None = None, k: int = 5) -> list[Hypothesis]:
        """Generate the top-k most novel directions worth exploring next.

        Combines: (a) idea-source hypotheses scored by novelty against coverage,
        and (b) explicit gap-filling from the self-model's known blind spots.
        """
        known = [(ln.domain, self.coverage._sig(ln.lesson)) for ln in engine.hub._learnings]
        domains = sorted(self.coverage.domains) or sorted({t.domain for t in engine.registry.all()})

        # two distinct exploration modes, each gets guaranteed room:
        #  - GAP frontiers: explore what the wheel KNOWS it doesn't know
        #  - NOVELTY frontiers: seek the unseen (transfer + mixing)
        gap_frontiers = [
            Hypothesis(
                description=f"investigate blind spot: {g}",
                target_domain="_meta", source="gap", novelty=0.95,
                rationale="the self-model flagged this as something it does not know",
            )
            for g in (gaps or [])
        ]

        novelty_cands = self.idea_source.generate(known, domains)
        for h in novelty_cands:
            h.novelty = self.coverage.novelty(h.target_domain, h.description)
        novelty_cands.sort(key=lambda h: -h.novelty)

        # reserve up to half the budget for gaps (when present), rest for novelty
        gap_slots = min(len(gap_frontiers), max(1, k // 2)) if gap_frontiers else 0
        out: list[Hypothesis] = list(gap_frontiers[:gap_slots])
        seen = {h.description for h in out}
        for h in novelty_cands:
            if len(out) >= k:
                break
            if h.description in seen:
                continue
            seen.add(h.description)
            out.append(h)
        # if novelty didn't fill the budget, top up with remaining gaps
        for h in gap_frontiers[gap_slots:]:
            if len(out) >= k:
                break
            out.append(h)
        return out
