# DistilBERT original feature set — multiseed protocol freeze

Protocol frozen before launching seed 2026. Seeds 42 and 123 were verified to
match every fixed field below; seed 2026 must differ only in the random seed and
run-owned output paths.

## Scientific purpose

Estimate training stochasticity for the within-dataset CICIoT2023 comparison.
This protocol supports mean/standard-deviation reporting, paired comparisons
against fixed classical predictions, and selection of a median-performing
checkpoint for XAI. It does not establish cross-dataset generalization.

## Fixed configuration

- Model: `distilbert-base-uncased`, eight categories.
- Feature set: `original` (39 serialized fields).
- Leakage-safe grouped split: train 42,853; validation 8,572; test 8,572.
- Serialized test SHA-256:
  `a209b9fdfe3152689a445954a7182819541ed9f20d27d611c7ae1a2829c196f8`.
- Seeds: 42, 123, 2026 (declared before observing seed-2026 results).
- Maximum epochs: 15; early-stopping patience: 4.
- Checkpoint selection: highest validation macro-F1 in FP32.
- Training: BF16 autocast on Apple MPS; validation and test: FP32.
- Batch size: 32; evaluation batch size: 64; maximum length: 224.
- Learning rate: 2e-5; linear warmup/decay; warmup fraction: 0.10.
- Test data are never used for checkpoint or stopping decisions.
- Every run must export aligned `SampleID` predictions and pass prediction-array
  and confusion-matrix reconstruction assertions.

## Code/environment freeze

- `scripts/04_finetune.py` SHA-256:
  `1d5b90af04682ec6fc9afa2ffd69a426d41a9fefb34d84cac862ad092f37dc29`.
- `scripts/pipeline_config.py` SHA-256:
  `1144d8f8b34af5d2bcbe4f2c6beb5f179ebc043ddf71cf2af6ae35c8988dbf26`.
- `requirements-lock.txt` SHA-256:
  `3975f66a7905c38859d4da810ed4e3a78aa2be78c68c73619a65c7b133c73abd`.

Do not modify these inputs until seed 2026 completes. Any unavoidable change
requires a new protocol version and rerunning all affected seeds.
