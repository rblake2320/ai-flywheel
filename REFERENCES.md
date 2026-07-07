# References

The design decisions in this engine are grounded in published research.

## Privacy & the membership-inference threat (why k-anonymity + tenant isolation)

- **Morris et al. (2025), "How Much Do Language Models Memorize?"** — arXiv:2505.24832.
  Measures GPT-style model capacity at **~3.6 bits per parameter**, cleanly separates
  *unintended memorization* from *generalization*, and derives **scaling laws relating
  model capacity + dataset size to membership inference**.

  Relevance here: membership inference — determining whether a specific record was in a
  model's training data — is precisely the leak that this engine's **k-anonymity gate**
  (`learning/hub.py`, a learning crosses only when ≥k similar exist) and **tenant
  isolation** (`tenancy/`, private content never crosses the boundary) defend against.
  The paper also bounds fine-tune leakage quantitatively: a small tenant model has a
  finite memorization budget (params × 3.6 bits), which caps both how much tenant-
  specific detail it can absorb and how much it can leak.

## Model collapse (why the RealDataFloor)

- **Shumailov et al. (2024), Nature** — model collapse from training on generative
  output; and **"Strong Model Collapse" (ICLR 2025)** — even a small synthetic fraction
  degrades performance. Basis for `core/provenance.py`'s "accumulate, don't replace"
  real-data floor.
