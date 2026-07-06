# Roadmap

Grounded in a 2025-2026 deep-dive across NVIDIA NeMo, model-collapse research,
multi-tenant privacy, and the Python ML stack. The core stays zero-dependency
and pure-Python; every heavy capability lands as an **optional plugin behind a
Protocol**, installed via extras.

## Shipped (v0.3.0) — hardened against the research

- **Model-collapse defense** (`core/provenance.py`): provenance tags on every
  interaction + a `RealDataFloor` that enforces "accumulate, don't replace" —
  a batch that is too synthetic is trimmed or refused before training. *(the
  #1 flag: Nature 2024 / ICLR 2025 "Strong Model Collapse")*
- **k-anonymity gate** (`learning/hub.py`): a learning distributes only once
  `min_k` similar learnings exist — turns "anonymized" into a threshold, not a
  hope. Defends against re-identification via unique-event quasi-identifiers.
- **Untrusted inbound**: learnings handed back to tenants are flagged
  `trusted=False` — closes the cross-tenant prompt-injection / poisoning channel.
- **Sanitizer hardening** (`safety/sanitizer.py`): Unicode NFKC + zero-width
  strip (homoglyph/hidden-char evasion), SSN + credit-card + grouped-phone
  regex, and a pluggable `PIIDetector` seam for contextual NER.
- **Real accelerometer** (`metrics/accelerometer.py`): marginal-value-per-batch
  (the true "is it stalling" signal) + diversity/coverage, not just mean reward.

## The boundary (read before any integration work)

**This engine is the learning organ, not the brain.** A separate governed-recall
"cortex" (budget-gated, provenance-chained, trust-promoted multi-store memory)
is a PRIVATE, patent-adjacent system. The ONLY things that may cross into this
public repo are: (1) the WhyCase outbox schema, and (2) a thin HTTP/MCP client to
the brain's `/recall`. No cortex logic, no event-log ownership, no second memory.
Same rule as pre-patent protocols: decide/file before anything else crosses.

## Shipped (v0.9.0) — self-explore + the organism (everything connects)

The Disney model in code: not stacked faculties, one connected complex run.

- **Self-explore** (`explore/curiosity.py`): open-ended, curiosity-driven
  exploration — the opposite pole of self-learn (targeted research). Grounded in
  real prior art (novelty search / MAP-Elites, Schmidhuber curiosity). A
  `CoverageMap` archives what's been seen; the `Explorer` seeks the NOVEL via
  cross-domain transfer (the Disney mechanism — carry a retail pattern into real
  estate) + pattern mixing, and turns the self-model's blind spots into
  exploration targets. Pluggable `IdeaSource` (combinatorial core; LLM/web plugs
  in for genuinely open-ended invention).
- **The organism** (`organism.py`): `run_cycle()` connects every faculty —
  self-model GAPS drive exploration, CONFIDENCE governs the explore/exploit
  balance (low confidence explores more; this is how self-learn and self-explore
  share ONE axis), FRONTIERS become next-cycle experiments, WhyCases feed
  curation. `aiflywheel organism` shows the connections firing.
- **Emergent safety, free**: explored hypotheses are SYNTHETIC, so the
  RealDataFloor automatically bounds how much curiosity can influence training —
  exploration that cannot collapse the model, with no separate limiter.

## Shipped (v0.8.0) — operational self-awareness (a self-model)

NOT sentience — a self-model in the engineering sense. `introspection/self_model.py`
`SelfModel` lets the engine reason about ITSELF, every method grounded in real
measured state (no invented confidence):

- **capabilities()** — what faculties are actually wired (not what's possible).
- **confidence()** — a computed self-trust score; LOW when data is thin,
  synthetic-heavy, or recently self-corrected. The wheel knowing when not to be sure.
- **known_gaps()** — its own blind spots stated plainly (no training yet, not
  networked, no golden set, recall limb degraded, collapse risk).
- **explain()** — why it self-corrected, from its own WhyCases.
- **self_check()** — the deepest bit: detects contradictions between what it
  believes about itself and what's actually true (decision-accounting mismatch,
  claims-networked-but-isn't, unrecorded rollbacks). A self-model that can catch
  itself being wrong.

## Shipped (v0.7.0) — brain-ready integration seams

- **EventLogSource** (`ingest/event_log.py`): tails the brain's append-only event
  log (JSONL, offset-tracked, partial-line tolerant) and maps events to
  Interactions. The flywheel CONSUMES the brain's single honest event source — it
  never opens its own capture path. Unknown-tenant events are skipped, not crashed.
- **Pluggable recall** (`reflection/recall.py`): `RecallProvider` protocol with
  `LocalRecall` (WhyStore bootstrap) and `RemoteRecall` (thin stdlib HTTP client
  to the brain's `/recall`, fails open). The engine prefers an injected provider,
  so the moment the brain's `/recall` exists the flywheel becomes its client —
  one recall system for the whole organism, never two.

## Shipped (v0.5.0–v0.6.0) — closed the loop, made it self-reflective

- **Loop closure** (`metrics/promotion.py`): snapshot → train → evaluate →
  PROMOTE/ROLLBACK. A regressing batch is reverted to the pre-train snapshot —
  the wheel *cannot make itself worse*. `Revertable` snapshot/rollback on all
  learners incl. the real TRL one.
- **Real reward source** (`core/reward.JudgeRewardVerifier`): an independent
  Judge scores answers instead of trusting the tenant's self-reported reward.
- **Self-reflection** (`reflection/whycase.py`): every rollback becomes a
  durable, recallable WhyCase in why-engine's schema. Before training the wheel
  can RECALL whether a similar batch regressed before — it learns from its own
  corrections instead of just undoing them.
- **Fact layer (no fakes)**: a WhyCase carries the ACTUAL measured numbers
  (real prev/new quality, batch, tenants); a "regression" case whose numbers
  don't show a real regression is REFUSED. Truth-checked, not filler.
- **Durable**: WhyCases are fsync'd atomic writes that reload on restart —
  a recorded lesson is never lost.

## Shipped (v0.4.0) — from hardened engine to best-in-class

- **Multi-stage Curator** (`curation/curator.py`): reward → semantic-dedup →
  diversity pipeline, replacing the scalar valve. Pure-Python Jaccard dedup +
  dominance cap; pluggable to SemHash/NeMo behind `CuratorStage`.
- **Judge + RegressionGate** (`metrics/judge.py`): win-rate over a held set and
  a frozen golden set that blocks silent backsliding. DeepEval plugs in behind
  the `Judge` Protocol.
- **Real `FewShotLearner`** (`core/learner.py`): not a stand-in — curates a
  top-N exemplar bank per domain and answers prompts, scorable by a Judge.
- **Per-tenant lift attribution** (`metrics/attribution.py`): gained vs
  contributed — the network-effect proof and pricing lever (demo shows ~2.0×).
- **Plugin discovery** (`plugins.py`): `entry_points` group so third parties
  add backends with zero core changes.
- **Runnable demo** (`aiflywheel demo`) + `did_accelerate`/`peak_quality` on the
  accelerometer.

## Next — plug in real backends (Protocol seams already defined)

| Capability | Plugin (extra) | Seam | Source of choice |
|---|---|---|---|
| Real training | TRL `SFT/DPO/GRPO` (`[trainer]`), Unsloth accel (`[unsloth]`) | `core.learner.Learner` | TRL is the lingua franca; Unsloth first-class on RTX 50-series |
| LLM-as-judge accelerometer | DeepEval (`[eval]`) | new `Judge` Protocol | Apache-2.0, pytest-native, self-hostable, model-agnostic G-Eval |
| Semantic dedup curation | SemHash + model2vec (`[dedup]`) | new `Deduper` stage | numpy-only, no torch; dedup per-tenant then cross-tenant |
| Contextual PII | Presidio (`[pii]`) | `safety.sanitizer.PIIDetector` | regex→NER layering; regex alone ~57-73% recall |
| Off-the-shelf reward model | Skywork-Reward-V2 (HF, runtime pull) | `core.reward.RewardVerifier` | fits a 5090 alongside the trainer; benchmark on RewardBench 2 |

## Later — the harder guarantees

- **Multi-stage pluggable Curator** valve: reward → SemDeDup → diversity/value
  (Data-Shapley; DPO makes valuation linear-cost, arXiv 2512.15765).
- **Differential privacy / secure aggregation** for cross-tenant learnings
  (Opacus, NVIDIA FLARE, Clio-style aggregation) — makes "anonymized" a formal
  guarantee, and fits the DGX Spark / confidential-GPU hardware.
- **Provenance/audit trail**: trace which learnings improved which model
  (influence functions) — the multi-tenant trust story.
- **Golden-set regression gate** + drift-triggered retraining (NVIDIA's
  periodic-or-on-drift pattern) so the wheel never silently backslides.
- **Engineering seams**: `entry_points`-based plugin discovery, OTel `gen_ai.*`
  spans, optional Redis-Streams ingestion / Huey queue for multi-process scale.

## Deliberately NOT reinventing (wrap as plugins)

GPU-scale curation (NeMo Curator / Data-Juicer), RL training (TRL / OpenPipe
ART), prompt optimization (DSPy/GEPA), synthetic data (distilabel), labeling &
tracing UIs (Label Studio + Langfuse/Phoenix). Our niche is the lightweight,
business-agnostic, multi-tenant orchestration core that delegates heavy stages
to best-of-breed engines through clean Protocols.
