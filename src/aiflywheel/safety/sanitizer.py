"""
LearningSanitizer — closes the cross_learning leak.

`cross_learning` is the one free-form field allowed to cross the tenant
boundary. That makes it the boundary's weakest point: a tenant could embed
proprietary content (emails, phone numbers, account/comp figures, a customer
name, a private code) in it and it would reach the shared hub.

This sanitizer scrubs a learning before it is EVER allowed to cross. It:
  - redacts emails, phone numbers, long digit runs, currency amounts,
    and any tenant-declared secret terms,
  - enforces a max length (learnings are lessons, not documents),
  - FAILS CLOSED: if after scrubbing the text is empty, or if a hard-block
    term survived, the learning is rejected (not shared).

A rejected learning does not stop the interaction from training locally; it
simply never enters the shared pool. Better a lost lesson than a leak.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?:\+?\d[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?(?:\d[\s.\-]?){6,}\d")
_CURRENCY = re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d+)?")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")       # card-like grouped digits
_LONG_DIGITS = re.compile(r"\b\d{5,}\b")            # ids, account numbers, codes
_ZERO_WIDTH = re.compile(r"[​-‏‪-‮﻿]")


@runtime_checkable
class PIIDetector(Protocol):
    """Optional pluggable detector for contextual PII (names/orgs/locations).

    Regex cannot see contextual PII; a real deployment plugs in Presidio or a
    transformer NER here (extra `ai-flywheel[pii]`). `detect` returns spans of
    text to redact. The default engine ships NO detector (regex-only core).
    """

    def detect(self, text: str) -> list[str]:
        """Return substrings that should be redacted."""
        ...


@dataclass
class SanitizeResult:
    ok: bool
    text: str
    redactions: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class LearningSanitizer:
    """Scrubs and validates a cross_learning before it may cross the boundary."""

    max_length: int = 280
    # terms that must never appear even after scrubbing (hard block, case-insens.)
    hard_block: frozenset[str] = frozenset()
    # tenant-declared secret substrings to redact (soft: redacted, not blocked)
    secret_terms: frozenset[str] = frozenset()
    # optional contextual-PII detector (Presidio/NER) — None = regex-only core
    detector: PIIDetector | None = None

    @staticmethod
    def _normalize(text: str) -> str:
        """NFKC-normalize + strip zero-width chars so homoglyph/hidden-char
        evasion of the term filters can't slip proprietary content through."""
        text = unicodedata.normalize("NFKC", text)
        return _ZERO_WIDTH.sub("", text)

    def sanitize(self, text: str | None) -> SanitizeResult:
        if not text or not text.strip():
            return SanitizeResult(ok=False, text="", reason="empty")

        redactions: list[str] = []
        scrubbed = self._normalize(text)

        # optional contextual PII (names/orgs/locations) — before regex
        if self.detector is not None:
            for span in self.detector.detect(scrubbed):
                if span:
                    scrubbed = scrubbed.replace(span, "[REDACTED]")
                    redactions.append("pii_ner")

        # tenant-declared secrets (before regex mangles them)
        for term in self.secret_terms:
            if term and term.lower() in scrubbed.lower():
                scrubbed = re.sub(re.escape(term), "[REDACTED]", scrubbed, flags=re.I)
                redactions.append("secret_term")

        for label, pat in (
            ("email", _EMAIL), ("ssn", _SSN), ("card", _CARD),
            ("phone", _PHONE), ("currency", _CURRENCY), ("long_digits", _LONG_DIGITS),
        ):
            if pat.search(scrubbed):
                scrubbed = pat.sub("[REDACTED]", scrubbed)
                redactions.append(label)

        # hard-block terms => reject outright, even if we could redact
        low = scrubbed.lower()
        for term in self.hard_block:
            if term and term.lower() in low:
                return SanitizeResult(
                    ok=False, text="", redactions=redactions,
                    reason=f"hard_block:{term}",
                )

        scrubbed = scrubbed.strip()[: self.max_length].strip()

        # fail closed: nothing meaningful left after scrubbing
        if not scrubbed or scrubbed.replace("[REDACTED]", "").strip() == "":
            return SanitizeResult(
                ok=False, text="", redactions=redactions, reason="empty_after_scrub"
            )

        return SanitizeResult(ok=True, text=scrubbed, redactions=redactions)
