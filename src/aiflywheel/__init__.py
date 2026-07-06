"""aiflywheel — a horizontal, multi-tenant, business-agnostic AI data flywheel."""
from aiflywheel.core.interaction import Interaction
from aiflywheel.core.learner import Learner, SimulatedLearner
from aiflywheel.engine import FlywheelEngine, SubmitResult
from aiflywheel.tenancy.tenant import IsolationError, IsolationGuard, Tenant, TenantRegistry

__version__ = "0.6.0"
__all__ = [
    "FlywheelEngine", "SubmitResult", "Interaction",
    "Tenant", "TenantRegistry", "IsolationGuard", "IsolationError",
    "Learner", "SimulatedLearner",
]
