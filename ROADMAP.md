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
