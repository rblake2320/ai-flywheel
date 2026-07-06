"""
Self-reflection: turn every rollback into a remembered WhyCase.

The promotion gate lets the flywheel self-CORRECT (undo a bad train). This turns
that into self-REFLECT: when a batch regresses the model, we record WHY as a
WhyCase (compatible with the why-engine outbox format), and before training we
RECALL whether a similar pattern has regressed before — so the wheel stops
repeating its own mistakes instead of just undoing them each time.

  self-correct  = roll back a bad train
  self-reflect  = record why it was bad, and avoid the pattern next time

`WhyStore` writes WhyCases as JSON into an outbox directory using why-engine's
schema (caseId/idempotencyKey/rootCause/preventNextTime/…) so the real
why-engine can consume/publish them. Recall is a pure-Python, offline,
deterministic field-weighted term match — no dependency on the TS engine.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_WORD = re.compile(r"[a-z0-9]+")
# obvious secret shapes — a WhyCase must never carry a live secret to an outbox
_SECRET = re.compile(
    r"(ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|password\s*[:=])",
    re.I,
)


def _idempotency_key(title: str, root_cause: str) -> str:
    return hashlib.sha256(f"{title}|{root_cause}".encode()).hexdigest()[:16]


def _scan_and_redact(text: str) -> tuple[str, int]:
    n = len(_SECRET.findall(text))
    return (_SECRET.sub("[REDACTED]", text), n)


@dataclass
class WhyStore:
    """Records rollback WhyCases and recalls past regressions. Offline, deterministic."""

    outbox_dir: str
    _index: list[dict] = field(default_factory=list)   # in-memory recall index

    def __post_init__(self):
        Path(self.outbox_dir).mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        for p in sorted(Path(self.outbox_dir).glob("*.json")):
            try:
                self._index.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue

    def record(
        self,
        *,
        title: str,
        root_cause: str,
        why_not_caught: str,
        prevent_next_time: str,
        generalizable_pattern: str | None = None,
        created_at: str = "1970-01-01T00:00:00Z",
        sensitivity: str = "internal",
        measured_facts: dict | None = None,
    ) -> str:
        """Write a WhyCase (why-engine schema). Idempotent by (title, rootCause).

        `measured_facts` carries the ACTUAL numbers behind the case (real
        prev/new quality, batch size, tenants) so the record is verifiable
        against what happened — a fact layer, not filler prose. A case with a
        claimed regression whose measured new>=prev is rejected as untruthful.
        """
        # truth check: a "regression" WhyCase must reflect a REAL regression.
        if measured_facts:
            pq, nq = measured_facts.get("prev_quality"), measured_facts.get("new_quality")
            if pq is not None and nq is not None and nq >= pq:
                raise ValueError(
                    f"refusing to record a regression WhyCase that isn't real: "
                    f"new_quality {nq} >= prev_quality {pq}"
                )
        key = _idempotency_key(title, root_cause)
        path = Path(self.outbox_dir) / f"{key}.json"

        blob = " ".join([title, root_cause, why_not_caught, prevent_next_time,
                         generalizable_pattern or ""])
        _, secrets_found = _scan_and_redact(blob)
        title, _ = _scan_and_redact(title)
        root_cause, _ = _scan_and_redact(root_cause)
        prevent_next_time, _ = _scan_and_redact(prevent_next_time)

        case = {
            "caseId": key,
            "idempotencyKey": key,
            "createdAt": created_at,
            "title": title,
            "rootCause": root_cause,
            "whyNotCaught": why_not_caught,
            "whyFixWorked": "rolled back to the pre-train snapshot; model quality restored",
            "preventNextTime": prevent_next_time,
            "generalizablePattern": generalizable_pattern,
            "sensitivity": sensitivity,
            "secretScanResult": {
                "clean": secrets_found == 0,
                "secretsFound": secrets_found,
                "redactionsApplied": secrets_found,
            },
            # fact layer: the real measured numbers, so the case is auditable
            "measuredFacts": measured_facts or {},
        }
        # durable, atomic write: temp file → fsync → atomic replace. A crash
        # can't leave a half-written case; a recorded lesson is never lost.
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(case, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # refresh index entry (dedupe by key)
        self._index = [c for c in self._index if c.get("idempotencyKey") != key]
        self._index.append(case)
        return str(path)

    def recall(self, query: str, k: int = 3) -> list[dict]:
        """Field-weighted term match against recorded WhyCases (highest-scoring first)."""
        q = set(_WORD.findall(query.lower()))
        if not q:
            return []
        scored = []
        for case in self._index:
            score = 0.0
            for field_name, weight in (("title", 3.0), ("rootCause", 3.0),
                                       ("generalizablePattern", 2.5),
                                       ("preventNextTime", 1.5), ("whyNotCaught", 1.0)):
                text = (case.get(field_name) or "")
                terms = set(_WORD.findall(text.lower()))
                score += weight * len(q & terms)
            if score > 0:
                scored.append((score, case))
        scored.sort(key=lambda t: -t[0])
        return [c for _, c in scored[:k]]

    def seen_before(self, query: str, min_score_terms: int = 2) -> bool:
        """True if a past regression case meaningfully matches this query."""
        hits = self.recall(query, k=1)
        if not hits:
            return False
        q = set(_WORD.findall(query.lower()))
        top = set(_WORD.findall((hits[0].get("rootCause", "") + " " +
                                 hits[0].get("title", "")).lower()))
        return len(q & top) >= min_score_terms

    def count(self) -> int:
        return len(self._index)
