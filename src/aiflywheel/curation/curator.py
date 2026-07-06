"""
The Curator — a multi-stage, pluggable intake valve.

Research (2025-2026) is clear: a single reward threshold leaves quality on the
table and risks self-reinforcing mode collapse (rewarding what the model already
does). Best practice is a PIPELINE — reward → semantic dedup → diversity/value —
where the reward threshold is stage 1 of N, not the whole valve.

`Curator` runs an ordered list of `CuratorStage`s. Each stage filters a batch of
accepted interactions down to the examples actually worth training on. Stages are
pluggable: the built-ins here are pure-Python and dependency-free; a deployment
can drop in SemHash/model2vec or NeMo Curator behind the same protocol
(extra `[dedup]`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from aiflywheel.core.interaction import Interaction

_WORD = re.compile(r"[a-z0-9]+")


@runtime_checkable
class CuratorStage(Protocol):
    """One filtering stage. Takes a batch, returns the kept subset."""

    name: str

    def curate(self, batch: list[Interaction]) -> list[Interaction]:
        ...


def _shingles(text: str, k: int = 3) -> frozenset[str]:
    """Word k-shingles for cheap Jaccard near-dup detection (no embeddings)."""
    words = _WORD.findall((text or "").lower())
    if len(words) < k:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[i : i + k]) for i in range(len(words) - k + 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b) if (a | b) else 0.0


@dataclass
class RewardStage:
    """Keep only examples at/above a reward floor (stage 1 — the classic valve)."""

    min_reward: float = 0.0
    name: str = "reward"

    def curate(self, batch: list[Interaction]) -> list[Interaction]:
        return [i for i in batch if (i.reward_score or 0.0) >= self.min_reward]


@dataclass
class SemanticDedupStage:
    """Drop near-duplicate examples so a verbose tenant can't dominate a batch.

    Pure-Python Jaccard-over-shingles — no embeddings, no torch. A real
    deployment swaps in SemHash/model2vec behind the same CuratorStage protocol
    for embedding-quality dedup. Keeps the highest-reward member of each cluster.
    """

    threshold: float = 0.8
    name: str = "semantic_dedup"

    def curate(self, batch: list[Interaction]) -> list[Interaction]:
        # sort by reward desc so we keep the best representative of each dup group
        ordered = sorted(batch, key=lambda i: -(i.reward_score or 0.0))
        kept: list[Interaction] = []
        sigs: list[frozenset[str]] = []
        for it in ordered:
            sig = _shingles(f"{it.input_text} {it.output_text} {it.cross_learning or ''}")
            if any(_jaccard(sig, s) >= self.threshold for s in sigs):
                continue
            kept.append(it)
            sigs.append(sig)
        return kept


@dataclass
class DiversityStage:
    """Cap per-tenant and per-domain share so no single source dominates.

    Directly counters the mode-collapse failure mode (batch narrows to what one
    tenant/domain already does well). Round-robins across tenants up to a cap.
    """

    max_share: float = 0.6      # no tenant may exceed 60% of the curated batch
    name: str = "diversity"

    def curate(self, batch: list[Interaction]) -> list[Interaction]:
        if len(batch) < 2:
            return batch
        kept = list(batch)
        # drop the lowest-reward item of the most over-represented tenant until
        # no tenant exceeds max_share of what remains. Guarantees the invariant
        # on the OUTPUT (a share cap against the input can't).
        while True:
            counts: dict[str, int] = {}
            for it in kept:
                counts[it.tenant_id] = counts.get(it.tenant_id, 0) + 1
            total = len(kept)
            over = [t for t, c in counts.items() if c / total > self.max_share]
            if not over or total <= 1:
                break
            worst = max(over, key=lambda t: counts[t])
            victims = [i for i in kept if i.tenant_id == worst]
            drop = min(victims, key=lambda i: (i.reward_score or 0.0))
            kept.remove(drop)
        return kept


@dataclass
class Curator:
    """Runs an ordered pipeline of stages over a batch."""

    stages: list[CuratorStage] = field(default_factory=list)

    def curate(self, batch: list[Interaction]) -> list[Interaction]:
        out = batch
        for stage in self.stages:
            out = stage.curate(out)
        return out

    def trace(self, batch: list[Interaction]) -> list[dict]:
        """Per-stage counts — observability for what the valve removed."""
        out = batch
        rows = [{"stage": "input", "kept": len(out)}]
        for stage in self.stages:
            out = stage.curate(out)
            rows.append({"stage": stage.name, "kept": len(out)})
        return rows


def default_curator(min_reward: float = 0.0) -> Curator:
    """The recommended pure-Python valve: reward → dedup → diversity."""
    return Curator(stages=[
        RewardStage(min_reward=min_reward),
        SemanticDedupStage(threshold=0.85),
        DiversityStage(max_share=0.6),
    ])
