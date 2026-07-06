"""
TRLLearner — a REAL fine-tuning backend behind the `Learner` protocol.

This is the plugin that turns the flywheel from "provably-correct engine" into
"actually trains models." It runs Hugging Face TRL SFT with a LoRA adapter
(peft) on each accepted, curated batch of interactions, incrementally improving
a base model. On an NVIDIA GPU (e.g. DGX Spark GB10) it trains on-device; it
falls back to CPU for tiny smoke runs.

Design rules honored:
  - Everything heavy (torch/trl/transformers/peft) is imported LAZILY inside
    methods, so importing aiflywheel stays dependency-free. Install with the
    `[trainer]` extra to use this.
  - It consumes exactly what the engine hands it: a list of `Interaction` with
    input_text (prompt) and output_text (completion). The engine has already
    isolation-checked, reward-validated, curated, and enforced the real-data
    floor — so by here the batch is safe and worth training on.
  - `quality()` reports a real signal: 1 - normalized final training loss,
    clamped to [0,1], so the Accelerometer sees genuine batch-over-batch change.
"""
from __future__ import annotations

import os
import tempfile

from aiflywheel.core.interaction import Interaction


class TRLLearner:
    """Incremental SFT+LoRA trainer implementing the Learner protocol."""

    def __init__(
        self,
        base_model: str = "HuggingFaceTB/SmolLM2-135M",
        output_dir: str | None = None,
        max_seq_length: int = 512,
        epochs: float = 1.0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 2e-4,
        min_reward: float = 0.0,
        preflight: bool = True,
        allow_cpu: bool = True,
    ) -> None:
        self.base_model = base_model
        self.output_dir = output_dir or os.path.join(tempfile.gettempdir(), "aiflywheel-trl")
        self.max_seq_length = max_seq_length
        self.epochs = epochs
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.min_reward = min_reward
        # preflight: real GPU allocation test before EVERY train (not just setup).
        # On GB10 unified memory, cuda.is_available()==True does NOT mean usable —
        # resident inference (a 120B shard, an Ollama model) can leave the flag
        # green while a real allocation OOMs. allow_cpu keeps tiny smoke runs
        # working where there's no GPU at all.
        self.preflight = preflight
        self.allow_cpu = allow_cpu
        self._quality = 0.0
        self._steps = 0
        self._adapter_path: str | None = None

    # --- Learner protocol ---
    def quality(self) -> float:
        return round(self._quality, 4)

    def snapshot(self):
        """Capture the promoted state so a regressing train can be undone.

        Restores the active-adapter pointer + quality; a rejected adapter's files
        stay on disk (harmless) but are never pointed at again.
        """
        return (self._adapter_path, self._quality, self._steps)

    def rollback(self, snap) -> None:
        self._adapter_path, self._quality, self._steps = snap

    def train(self, batch: list[Interaction]) -> float:
        rows = self._to_rows(batch)
        if not rows:
            return self.quality()
        loss = self._run_sft(rows)
        # map final loss -> a bounded quality signal (lower loss = higher quality)
        self._quality = max(0.0, min(1.0, 1.0 / (1.0 + loss)))
        self._steps += len(rows)
        return self.quality()

    # --- helpers ---
    def _to_rows(self, batch: list[Interaction]) -> list[dict]:
        """Interaction -> SFT `text` rows. Plain-text format needs no chat
        template, so it works with ANY causal base model (not just Instruct
        variants). Uses only accepted, on-reward, non-empty examples."""
        rows = []
        for it in batch:
            if (it.reward_score or 0.0) < self.min_reward:
                continue
            if not it.input_text or not it.output_text:
                continue
            rows.append({
                "text": f"### Prompt:\n{it.input_text}\n\n### Response:\n{it.output_text}"
            })
        return rows

    def _preflight_cuda(self) -> str:
        """Real allocation test before training. Returns the device to use.

        cuda.is_available() is necessary but NOT sufficient on GB10 unified
        memory — GPU and system RAM share one pool, so resident inference can
        leave the flag green while a real allocation OOMs. We do an actual small
        allocation; if it fails we raise a clear error (naming the likely cause)
        instead of letting the trainer crash mid-run or trip the OOM-killer
        against a production inference server.
        """
        import torch

        if not torch.cuda.is_available():
            if self.allow_cpu:
                return "cpu"
            raise RuntimeError("CUDA not available and allow_cpu=False")
        try:
            probe = torch.zeros(1024, 512, device="cuda")  # ~2MB real allocation
            del probe
            torch.cuda.empty_cache()
            return "cuda"
        except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
            raise RuntimeError(
                "CUDA reports available but a 2MB probe allocation FAILED — the GPU "
                "is memory-contested (e.g. resident inference on GB10 unified memory). "
                "Target a free GPU (e.g. Spark2) or wait for memory to free. "
                f"underlying: {e}"
            ) from e

    def _run_sft(self, rows: list[dict]) -> float:
        # lazy, heavy imports — only when actually training
        import torch  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer

        if self.preflight:
            self._preflight_cuda()   # fail fast + safe if the GPU is contested

        resume = self._adapter_path  # continue from prior adapter if present
        ds = Dataset.from_list(rows)
        cfg = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=min(4, len(rows)),
            learning_rate=self.learning_rate,
            max_length=self.max_seq_length,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            bf16=_bf16_ok(),
        )
        peft_cfg = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
        )
        trainer = SFTTrainer(
            model=resume or self.base_model,
            args=cfg,
            train_dataset=ds,
            peft_config=peft_cfg if resume is None else None,
        )
        result = trainer.train()
        trainer.save_model(self.output_dir)
        self._adapter_path = self.output_dir
        return float(getattr(result, "training_loss", 0.0) or 0.0)


def _bf16_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001
        return False
