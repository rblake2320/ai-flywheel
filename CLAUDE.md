# CLAUDE.md — ai-flywheel

This repo is the **reference implementation** of a horizontal, multi-tenant,
business-agnostic AI data flywheel engine (package `aiflywheel`, under `src/`).
It also keeps `FLYWHEEL.md`, the architecture reference. **PUBLIC repo.**

## What this repo is

- A real, runnable Python engine — the business-agnostic CORE of the flywheel.
  Vertical copilots (MK Copilot & siblings) are TENANTS that plug in; they are
  never part of the core. See `README.md` for the model.
- Because it is business-agnostic, it is safe to be public: **no tenant's
  proprietary data, credentials, or business specifics ever belong here.** That
  is enforced in code by the tenant boundary (`core/interaction.py` +
  `tenancy/tenant.py`), and it is also a hard rule for this repo's contents.
- `FLYWHEEL.md` remains the architecture doc. The larger private `ai-business`
  project (DGX Sparks) holds the tenant-specific systems this engine serves.

## Commands

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
```

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
