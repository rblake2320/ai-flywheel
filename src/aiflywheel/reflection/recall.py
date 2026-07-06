"""
RecallProvider — one recall system, not two.

The flywheel's local WhyStore recall is a BOOTSTRAP. The architectural rule is:
the moment the brain's governed /recall exists, the flywheel becomes a CLIENT of
it — two recall systems would be day-one fragmentation. This abstraction makes
that a one-line swap: `LocalRecall` (wraps the WhyStore) today, `RemoteRecall`
(HTTP client to the brain) tomorrow. Nothing else in the engine changes.

Boundary discipline: RemoteRecall is a thin HTTP client (stdlib urllib only) —
it carries NO cortex logic. The governed, provenance-chained, trust-promoted
recall lives in the brain's PRIVATE repo. Only the query/response crosses here.
Fails OPEN: if the brain is unreachable, recall returns "not seen" so the
flywheel never blocks on it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib import request


@runtime_checkable
class RecallProvider(Protocol):
    """Answers 'have we seen something like this before?'"""

    def seen_before(self, query: str) -> bool:
        ...

    def recall(self, query: str, k: int = 3) -> list[dict]:
        ...


@dataclass
class LocalRecall:
    """Bootstrap recall over the flywheel's own WhyStore (offline, deterministic)."""

    store: object  # reflection.whycase.WhyStore

    def seen_before(self, query: str) -> bool:
        return bool(self.store.seen_before(query))

    def recall(self, query: str, k: int = 3) -> list[dict]:
        return self.store.recall(query, k=k)


@dataclass
class RemoteRecall:
    """Client of the brain's governed /recall. Thin HTTP; no cortex logic here."""

    base_url: str                       # e.g. http://localhost:PORT
    timeout: float = 2.0
    min_hits: int = 1

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = request.Request(
            self.base_url.rstrip("/") + path, data=data,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode())

    def recall(self, query: str, k: int = 3) -> list[dict]:
        try:
            out = self._post("/recall", {"query": query, "k": k})
            return out.get("results", []) if isinstance(out, dict) else []
        except Exception:  # noqa: BLE001 - fail open: brain down != flywheel down
            return []

    def seen_before(self, query: str) -> bool:
        return len(self.recall(query, k=self.min_hits)) >= self.min_hits
