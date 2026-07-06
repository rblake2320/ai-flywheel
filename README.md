# ai-flywheel

**A horizontal, multi-tenant, business-agnostic AI data flywheel engine.**

Vertical business copilots plug in as *isolated tenants*. Each contributes only
anonymized learnings upward and receives improved shared models downward — so
every copilot gets smarter from the data of all the others, **without any
tenant's proprietary data ever crossing the boundary.**

This is the reference implementation of the corrected flywheel architecture:
the engine is the shared, business-agnostic core; the businesses (an MK/Mary-Kay
seller copilot, a real-estate copilot, a car-sales copilot…) are tenants of it,
never part of it.

---

## The two ideas that make it "the best flywheel"

1. **A hard tenant boundary.** An `Interaction` splits into *private* fields
   (raw input/output/prompt — never leave the tenant) and *shareable* fields
   (reward signal, anonymized `cross_learning`). `to_shared()` is the only path
   across, and an `IsolationGuard` fails **closed** on any leak. Proprietary
   data cannot pollute the shared brain.

2. **A measured network effect.** The `CrossLearningHub` pools each tenant's
   anonymized learnings and redistributes them to all others, and the
   `Accelerometer` proves the loop is actually turning — reporting
   `ACCELERATING` / `STEADY` / `STALLING` and whether the latest gains are
   `networked` (drew on more than one tenant). A flywheel you can't measure is
   just a wheel.

Plus a **self-tuning intake valve** (`AdaptiveThreshold`) that floats to hold a
target accept rate instead of a hardcoded cutoff.

---

## How it works — the full loop

```
tenant.report(interaction)
   → IsolationGuard redacts to shareable-only     (tenant boundary)
   → AdaptiveThreshold scores + decides accept     (self-tuning valve)
   → accepted → training queue + cross_learning → CrossLearningHub  (network effect)
   → batch fills → Learner.train() → better model
   → Accelerometer records batch quality           (is it speeding up?)
   → tenant.sync() pulls improved model + others' anonymized learnings
```

The engine ships a deterministic `SimulatedLearner` so the whole loop runs and
is tested with zero external dependencies. A production tenant supplies a real
`Learner` (SFT/DPO on GPUs, NeMo, etc.) behind the same one-method interface.

---

## Quickstart

```python
from aiflywheel import FlywheelEngine, Tenant
from aiflywheel.contract.sdk import FlywheelClient

engine = FlywheelEngine(batch_size=50)
engine.add_tenant(Tenant(tenant_id="mk-copilot", domain="retail"))
engine.add_tenant(Tenant(tenant_id="realty-bot", domain="real_estate"))

mk = FlywheelClient(engine, "mk-copilot")
mk.report(
    input_text="…Mary Kay seller's private question…",   # stays local
    output_text="…private answer…",                       # stays local
    reward=0.92,
    domain="retail",
    cross_learning="objection-handling pattern X lifts conversion",  # anonymized, shared
)

update = mk.report and mk.sync()   # pull improved model + OTHER tenants' learnings
print(engine.health())             # is the flywheel turning?
```

## Install & test

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q          # 15 tests, incl. the network-effect proof
```

## Layout

```
src/aiflywheel/
  core/interaction.py   the atomic unit + the tenant boundary (to_shared)
  core/learner.py       pluggable trainer protocol + SimulatedLearner
  tenancy/tenant.py     Tenant, TenantRegistry, IsolationGuard (fail-closed)
  adaptive/threshold.py self-tuning accept-rate valve
  learning/hub.py       CrossLearningHub — the network effect
  metrics/accelerometer.py  is-it-actually-accelerating metric
  contract/sdk.py       FlywheelClient — the tenant plug-in SDK
  engine.py             wires the full loop
```

## Status

v0.1.0 — working core, 15 passing tests, lint-clean. Business-agnostic and
open by design: no tenant ever puts proprietary data here. Roadmap: pluggable
real learners, persistence, per-tenant reward models, richer anonymization.

MIT.
