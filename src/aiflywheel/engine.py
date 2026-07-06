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
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aiflywheel.adaptive.threshold import AdaptiveThreshold
from aiflywheel.core.interaction import Interaction
from aiflywheel.core.learner import Learner, SimulatedLearner
from aiflywheel.core.reward import RewardTracker, RewardVerifier, clamp_reward
from aiflywheel.learning.hub import CrossLearningHub, SharedLearning
from aiflywheel.metrics.accelerometer import Accelerometer
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
    verifier: RewardVerifier | None = None
    _queue: list[Interaction] = field(default_factory=list)

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
            else:
                result.reject_reason = clean.reason

        # 5) batch full → train, measure acceleration, re-tune threshold
        if len(self._queue) >= self.batch_size:
            self._flush()
            result.trained = True
            result.model_quality = self.learner.quality()

        return result

    def _flush(self) -> None:
        batch = self._queue
        self._queue = []
        mean_reward = sum((i.reward_score or 0.0) for i in batch) / len(batch)
        n_sources = len({i.tenant_id for i in batch})
        self.learner.train(batch)
        self.accel.record(mean_reward, len(batch), n_sources)
        self.threshold.update()

    def flush(self) -> None:
        """Force-train whatever is queued (e.g. at shutdown)."""
        if self._queue:
            self._flush()

    # --- what a consuming tenant pulls back down ---
    def pull(self, tenant_id: str) -> dict:
        """Improved model quality + cross-tenant learnings for a consumer."""
        tenant = self.registry.get(tenant_id)
        learnings = self.hub.distribute(tenant_id) if tenant.consumes else []
        return {
            "model_quality": self.learner.quality(),
            "cross_learnings": learnings,
            "acceleration": self.accel.report(),
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
        }

    # --- durability ---
    def save(self, path: str) -> None:
        """Persist the flywheel's shareable momentum (hub + accelerometer)."""
        from aiflywheel.persistence.store import FlywheelStore

        FlywheelStore(path).save(self.hub, self.accel)

    def load(self, path: str) -> None:
        """Rehydrate hub + accelerometer so the wheel keeps its momentum."""
        from aiflywheel.persistence.store import FlywheelStore

        FlywheelStore(path).load(self.hub, self.accel)
