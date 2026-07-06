"""
Judge + golden-set regression gate — the real "did it get better" signal.

Self-reported model quality is not enough. Real systems (2025-2026) measure a
new model two ways before promoting it:

  1. WIN RATE — an independent judge prefers the new model over the previous one
     on a held prompt set (AlpacaEval-style). This is the primary acceleration
     signal.
  2. REGRESSION GATE — the new model must NOT backslide on a FROZEN golden set
     of known-good cases. A flywheel can improve on average while silently
     breaking things it used to get right; the gate blocks that.

`Judge` is a Protocol: the default `DeterministicJudge` needs no LLM (for tests
and offline runs); a deployment plugs in DeepEval / an LLM-as-judge behind the
same interface (extra `[eval]`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    """Scores a model's answer to a prompt in [0,1]. No content leaves the host."""

    def score(self, prompt: str, answer: str) -> float:
        ...


@dataclass
class GoldenCase:
    prompt: str
    min_score: float = 0.6      # the frozen bar this case must keep clearing


@dataclass
class DeterministicJudge:
    """A no-LLM judge: scores by a supplied scoring function. For tests/offline.

    Real deployments replace this with an LLM-as-judge (DeepEval G-Eval) behind
    the same `Judge` protocol; nothing else in the pipeline changes.
    """

    scorer: object = None       # callable(prompt, answer) -> float

    def score(self, prompt: str, answer: str) -> float:
        if self.scorer is None:
            return 1.0 if answer else 0.0
        return max(0.0, min(1.0, float(self.scorer(prompt, answer))))


@dataclass
class RegressionGate:
    """A frozen golden set the model must never backslide on."""

    cases: list[GoldenCase] = field(default_factory=list)

    def evaluate(self, judge: Judge, answer_fn) -> dict:
        """answer_fn(prompt) -> answer. Returns pass/fail + per-case regressions."""
        if not self.cases:
            return {"passed": True, "regressions": [], "n": 0, "mean": None}
        regressions, total = [], 0.0
        for case in self.cases:
            s = judge.score(case.prompt, answer_fn(case.prompt))
            total += s
            if s < case.min_score:
                regressions.append({"prompt": case.prompt, "score": round(s, 4),
                                    "required": case.min_score})
        return {
            "passed": len(regressions) == 0,
            "regressions": regressions,
            "n": len(self.cases),
            "mean": round(total / len(self.cases), 4),
        }


def win_rate(judge: Judge, prompts: list[str], new_fn, prev_fn) -> dict:
    """Fraction of prompts where the new model's answer beats the previous one."""
    if not prompts:
        return {"win_rate": None, "wins": 0, "losses": 0, "ties": 0, "n": 0}
    wins = losses = ties = 0
    for p in prompts:
        ns = judge.score(p, new_fn(p))
        ps = judge.score(p, prev_fn(p))
        if ns > ps:
            wins += 1
        elif ns < ps:
            losses += 1
        else:
            ties += 1
    n = len(prompts)
    return {"win_rate": round(wins / n, 4), "wins": wins, "losses": losses,
            "ties": ties, "n": n}
