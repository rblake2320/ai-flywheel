"""
Model-collapse defense: provenance + the real-data floor.

Training generatively on model output creates a degenerative loop — diversity
collapses, tails vanish, biases amplify generation-over-generation (Shumailov
et al., Nature 2024; "Strong Model Collapse", ICLR 2025). The empirically robust
mitigation is ACCUMULATE, don't REPLACE: always keep real/human data in the mix
and never let a training batch be dominated by synthetic/model-generated data.

`RealDataFloor` enforces exactly that. Before a batch trains, it checks the
fraction of non-synthetic examples against a floor. If the batch is too
synthetic, training is refused (or the batch is down-filtered to synthetic-safe
size) — the single cheapest insurance against collapse, and near-impossible to
retrofit once a collapsed model has poisoned the loop.
"""
from __future__ import annotations

from dataclasses import dataclass

from aiflywheel.core.interaction import Interaction

# provenance values considered "real" (safe, non-degenerative) vs synthetic.
REAL = frozenset({"real", "human"})
SYNTHETIC = frozenset({"synthetic", "model"})


def real_fraction(batch: list[Interaction]) -> float:
    if not batch:
        return 1.0
    real = sum(1 for i in batch if (i.provenance or "real") in REAL)
    return real / len(batch)


@dataclass
class RealDataFloor:
    """Refuses to train a batch that is too synthetic to be safe."""

    min_real_fraction: float = 0.5      # >= half of every batch must be real

    def check(self, batch: list[Interaction]) -> bool:
        """True if the batch is safe to train on."""
        return real_fraction(batch) >= self.min_real_fraction

    def enforce(self, batch: list[Interaction]) -> list[Interaction]:
        """Return a batch that satisfies the floor.

        If already safe, returns it unchanged. Otherwise trims synthetic
        examples until the floor is met (keeps all real ones), rather than
        training on a collapse-prone mix. May return fewer examples — that is
        the point: better a smaller real batch than a poisoned model.
        """
        if self.check(batch):
            return batch
        real = [i for i in batch if (i.provenance or "real") in REAL]
        synth = [i for i in batch if (i.provenance or "real") in SYNTHETIC]
        if not real:
            return []  # all synthetic → refuse entirely
        # keep all real; admit synthetic only up to the floor ratio
        # real / (real + k) >= f  ->  k <= real*(1-f)/f
        allowed_synth = int(len(real) * (1 - self.min_real_fraction) / self.min_real_fraction)
        return real + synth[:allowed_synth]
