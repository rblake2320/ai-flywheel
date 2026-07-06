#!/usr/bin/env python3
"""
Spark training worker — runs the TRLLearner ON the DGX Spark (GB10 GPU).

The flywheel engine (anywhere) curates a batch and writes it as JSONL; this
worker reads that JSONL on the Spark, runs a real SFT+LoRA step via TRLLearner,
and writes back a small JSON result (quality + adapter path). This is the seam
that lets the portable engine delegate heavy training to the GPU box.

Usage (on the Spark):
    python3 spark_train_worker.py --batch batch.jsonl --model <hf-model> \
        --out result.json --adapter-dir ~/aiflywheel-adapters/tenantX

Batch JSONL: one interaction per line with at least input_text, output_text,
reward_score (the engine's persistence / a simple dump produces this).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiflywheel.backends.trl_learner import TRLLearner  # noqa: E402
from aiflywheel.core.interaction import Interaction  # noqa: E402


def load_batch(path: str) -> list[Interaction]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out.append(Interaction(
            id=rec.get("id", ""), tenant_id=rec.get("tenant_id", "t"),
            timestamp=rec.get("timestamp", ""),
            input_text=rec.get("input_text", ""),
            output_text=rec.get("output_text", ""),
            reward_score=rec.get("reward_score"),
            domain=rec.get("domain"),
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    args = ap.parse_args()

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        device = "unknown"

    batch = load_batch(args.batch)
    learner = TRLLearner(base_model=args.model, output_dir=args.adapter_dir, epochs=args.epochs)
    quality = learner.train(batch)

    result = {
        "quality": quality,
        "n_examples": len(batch),
        "device": device,
        "adapter_path": learner._adapter_path,
        "model": args.model,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
