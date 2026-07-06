"""
FlywheelEngine — the horizontal core that ties every piece together.

The full loop, per the corrected architecture:

    tenant submits interaction
        → IsolationGuard redacts to a shareable record   (tenant boundary)
        → reward score observed by AdaptiveThreshold      (self-tuning valve)
        → if accepted: queued for training + its cross_learning contributed
          to the CrossLearningHub                          (network effect)
        → when a batch fills: Learner.train() produces a better model
        → Accelerometer records batch quality              (is it speeding up?)
        → improved model + pooled learnings flow back to all consuming tenants

The engine is business-agnostic. Tenants (vertical copilots) are the only
business-specific parts, and they only ever touch it through submit()/pull().

CONCURRENCY: a FlywheelEngine assumes a SINGLE WRITER — one process calling
submit()/flush(). It is not internally locked. With a live tenant feeding events
AND a separate training worker, funnel writes through one owner (e.g. one ingest
loop) or add external locking before wiring the second writer. Reads (health,
pull) are safe alongside a single writer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aiflywheel.adaptive.threshold import AdaptiveThreshold
from aiflywheel.core.interaction import Interaction
from aiflywheel.core.learner import Learner, SimulatedLearner
from aiflywheel.core.provenance import RealDataFloor, real_fraction
from aiflywheel.core.reward import RewardTracker, RewardVerifier, clamp_reward
from aiflywheel.curation.curator import Curator
from aiflywheel.learning.hub import CrossLearningHub, SharedLearning
from aiflywheel.metrics.accelerometer import Accelerometer
from aiflywheel.metrics.attribution import LiftLedger
from aiflywheel.metrics.promotion import ROLLBACK, PromotionGate
from aiflywheel.safety.sanitizer import LearningSanitizer
from aiflywheel.tenancy.tenant import IsolationGuard, Tenant, TenantRegistry


@dataclass
class SubmitResult:
    accepted: bool
    threshold: float
    trained: bool = False
    model_quality: float | None = None
    reward: float | None = None          # the reward actually used (post-validation)
    shared: bool = False                  # did a sanitized learning cross to the hub?
    reject_reason: str = ""               # if a learning was blocked from sharing


@dataclass
class FlywheelEngine:
    """The multi-tenant flywheel core."""

    batch_size: int = 50
    learner: Learner = field(default_factory=SimulatedLearner)
    threshold: AdaptiveThreshold = field(default_factory=AdaptiveThreshold)
    hub: CrossLearningHub = field(default_factory=CrossLearningHub)
    accel: Accelerometer = field(default_factory=Accelerometer)
    registry: TenantRegistry = field(default_factory=TenantRegistry)
    guard: IsolationGuard = field(default_factory=IsolationGuard)
    sanitizer: LearningSanitizer = field(default_factory=LearningSanitizer)
    rewards: RewardTracker = field(default_factory=RewardTracker)
    floor: RealDataFloor = field(default_factory=RealDataFloor)
    curator: Curator | None = None          # optional multi-stage intake valve
    lift: LiftLedger = field(default_factory=LiftLedger)
    promotion: PromotionGate | None = None  # optional loop-closing promote/rollback gate
    reflector: object | None = None         # optional WhyStore: self-reflection on rollback
    recall: object | None = None            # optional RecallProvider: brain's /recall client
    verifier: RewardVerifier | None = None
    _queue: list[Interaction] = field(default_factory=list)
    _promotions: int = 0
    _rollbacks: int = 0

    # --- tenant lifecycle ---
    def add_tenant(self, tenant: Tenant) -> Tenant:
        return self.registry.register(tenant)

    # --- the core loop ---
    def submit(self, interaction: Interaction) -> SubmitResult:
        """A tenant submits one interaction. Enforces isolation, scores, learns."""
        tenant = self.registry.get(interaction.tenant_id)  # KeyError if unknown

        # 1) tenant boundary: redact to shareable-only (raises on leak)
        self.guard.check_outbound(interaction)

        # 2) reward validation: never trust a raw tenant score
        score = clamp_reward(interaction.reward_score)
        if score is None:
            score = 0.0
        if self.verifier is not None:
            score = clamp_reward(self.verifier.verify(tenant.tenant_id, score)) or 0.0
        interaction.reward_score = score
        self.rewards.record(tenant.tenant_id, score)

        # 3) self-tuning quality valve
        accepted = self.threshold.observe(score)
        result = SubmitResult(
            accepted=accepted, threshold=self.threshold.value, reward=score
        )
        if not accepted:
            return result

        # 4) accepted → queue for training + contribute a SANITIZED learning
        self._queue.append(interaction)
        if tenant.contributes and interaction.cross_learning:
            clean = self.sanitizer.sanitize(interaction.cross_learning)
            if clean.ok:
                self.hub.contribute(
                    SharedLearning(
                        source_tenant=tenant.tenant_id,
                        domain=interaction.domain or tenant.domain,
                        lesson=clean.text,
                        reward_score=score,
                    )
                )
                result.shared = True
                self.lift.record_contribution(tenant.tenant_id)
            else:
                result.reject_reason = clean.reason

        # 5) batch full → train, measure acceleration, re-tune threshold
        if len(self._queue) >= self.batch_size:
            self._flush()
            result.trained = True
            result.model_quality = self.learner.quality()

        return result

    def _flush(self) -> None:
        raw = self._queue
        self._queue = []
        # multi-stage intake valve (reward → dedup → diversity), if configured.
        if self.curator is not None:
            raw = self.curator.curate(raw)
        # model-collapse defense: enforce the real-data floor BEFORE training.
        batch = self.floor.enforce(raw)
        if not batch:
            # entire batch too synthetic to be safe — refuse, don't poison.
            self.threshold.update()
            return
        mean_reward = sum((i.reward_score or 0.0) for i in batch) / len(batch)
        n_sources = len({i.tenant_id for i in batch})
        n_domains = len({(i.domain or "") for i in batch})

        # CLOSE THE LOOP: snapshot → train → evaluate → promote/rollback.
        prev_quality = self.learner.quality()
        snap = self.learner.snapshot() if hasattr(self.learner, "snapshot") else None
        self.learner.train(batch)
        new_quality = self.learner.quality()

        if self.promotion is not None:
            answer_fn = getattr(self.learner, "answer", None)
            # wrap answer(domain) as answer(prompt) for the golden set if present
            wrapped = (lambda p: answer_fn(p)) if answer_fn else None
            decision = self.promotion.decide(prev_quality, new_quality, answer_fn=wrapped)
            if decision.action == ROLLBACK and snap is not None:
                self.learner.rollback(snap)
                self._rollbacks += 1
                new_quality = self.learner.quality()
                self._reflect_on_rollback(decision, batch)
            else:
                self._promotions += 1

        self.accel.record(
            mean_reward, len(batch), n_sources,
            model_quality=new_quality,
            n_domains=n_domains,
            real_fraction=real_fraction(batch),
        )
        self.threshold.update()

    def _reflect_on_rollback(self, decision, batch: list[Interaction]) -> None:
        """Self-reflect: record WHY this batch regressed as a WhyCase, so the
        wheel can recall and avoid the pattern instead of just undoing it."""
        if self.reflector is None:
            return
        tenants = sorted({i.tenant_id for i in batch})
        domains = sorted({(i.domain or "?") for i in batch})
        pattern = f"regression from domains={domains} tenants={tenants}"
        try:
            self.reflector.record(
                title=f"training regression ({decision.reason})",
                root_cause=(
                    f"batch from tenants {tenants} in domains {domains} lowered model "
                    f"quality {decision.prev_quality}->{decision.new_quality}"
                ),
                why_not_caught=(
                    "passed the intake valve + real-data floor but degraded the model; "
                    "preventive filters can't see downstream quality — only the "
                    "promotion gate caught it after training"
                ),
                prevent_next_time=(
                    f"raise scrutiny for batches matching: {pattern}; consider tighter "
                    "dedup/diversity or a higher reward bar for these sources"
                ),
                generalizable_pattern=pattern,
                measured_facts={
                    "prev_quality": decision.prev_quality,
                    "new_quality": decision.new_quality,
                    "batch_size": len(batch),
                    "tenants": tenants,
                    "domains": domains,
                },
            )
        except Exception:  # noqa: BLE001 - reflection must never break the loop
            pass

    def seen_regression_before(self, batch: list[Interaction]) -> bool:
        """Recall: has a batch like this regressed the model before?

        Prefers an injected RecallProvider (the brain's governed /recall — one
        recall system for the whole organism); falls back to the local WhyStore
        as a bootstrap only when no provider is wired.
        """
        domains = sorted({(i.domain or "?") for i in batch})
        tenants = sorted({i.tenant_id for i in batch})
        query = f"regression domains {domains} tenants {tenants}"
        if self.recall is not None:
            return bool(self.recall.seen_before(query))
        if self.reflector is not None:
            return bool(self.reflector.seen_before(query))
        return False

    def flush(self) -> None:
        """Force-train whatever is queued (e.g. at shutdown)."""
        if self._queue:
            self._flush()

    # --- what a consuming tenant pulls back down ---
    def pull(self, tenant_id: str) -> dict:
        """Improved model quality + cross-tenant learnings for a consumer."""
        tenant = self.registry.get(tenant_id)
        learnings = self.hub.distribute(tenant_id) if tenant.consumes else []
        self.lift.record_gain(tenant_id, len(learnings))
        return {
            "model_quality": self.learner.quality(),
            "cross_learnings": learnings,
            "acceleration": self.accel.report(),
            "lift": self.lift.lift(tenant_id),
        }

    def health(self) -> dict:
        """One call that says whether the flywheel is actually turning."""
        return {
            "tenants": len(self.registry.all()),
            "threshold": self.threshold.value,
            "queue_depth": len(self._queue),
            "hub": self.hub.coverage(),
            "acceleration": self.accel.report(),
            "model_quality": self.learner.quality(),
            "promotions": self._promotions,
            "rollbacks": self._rollbacks,
            "why_cases": self.reflector.count() if self.reflector is not None else 0,
            # visible recall health — a down brain is counted, not silent
            "recall_fallbacks": getattr(self.recall, "fallbacks", 0),
        }

    # --- durability ---
    def save(self, path: str) -> None:
        """Durably persist ALL learned state (hub, accelerometer, threshold,
        counters, lift ledger, trust stats) — atomic + fsync'd."""
        from aiflywheel.persistence.store import FlywheelStore

        FlywheelStore(path).save(self)

    def load(self, path: str) -> None:
        """Rehydrate the full learned state so the wheel keeps its momentum."""
        from aiflywheel.persistence.store import FlywheelStore

        FlywheelStore(path).load(self)
