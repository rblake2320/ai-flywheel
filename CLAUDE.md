# CLAUDE.md — ai-flywheel

This repo holds one authoritative document: `FLYWHEEL.md`, the single source
of truth for the AI Army flywheel architecture (how the various product
copilots and specialists feed data back into a shared flywheel so each one
gets smarter from the others' data). **Private repo.**

## What this repo is

- A docs-only reference repo. It does not contain code, credentials, or the
  actual running systems.
- The real infrastructure and code this document describes live in the
  `ai-business` project (on the DGX Spark machines / the `Ai-Army` GitHub
  repo) — this repo is the portable, always-reachable copy of the
  architecture doc itself, so it can be read/updated from any machine without
  wading through that much larger, noisier repo.

## Canonical vs. speculative content — read this before trusting a claim

- `FLYWHEEL.md` is the sober, structured architecture reference. Treat it as
  current as of its "Last Updated" date, not as guaranteed present-tense fact —
  verify against the live systems before acting on anything time-sensitive.
- Separately, there is other material elsewhere (e.g. archived chat logs in
  `ai-business/shared/chat/archive/`) making much bigger claims — trillion-dollar
  valuations, "Disney of intelligence," acquisition talk. That material is an
  AI agent's own excited narrative, not a verified business case. Do not import
  those claims into this repo or treat them as established fact just because
  they reference the same "flywheel" concept.

## Editing etiquette

- Keep `FLYWHEEL.md` accurate and dated — update "Last Updated" when the
  architecture actually changes, based on verified system state, not vibes.
- No credentials, tokens, or live IPs belong here even though the repo is
  private — use placeholders if a path/host needs to be referenced generically.
- Small, verifiable edits over sweeping rewrites.
