# Macro-F1 bootstrap and baseline seed variance (reviewer #2)

Test-set bootstrap (4000 resamples, n=9000).

## Per-model macro-F1 (point estimate and bootstrap 95% CI)
- XGBoost: 0.7344  95% CI [0.7198, 0.7479]
- RF: 0.7258  95% CI [0.7105, 0.7399]
- DistilBERT: 0.7075  95% CI [0.6933, 0.7211]

## Paired macro-F1 difference (DistilBERT − baseline)
- DistilBERT − XGBoost: -0.0269  95% CI [-0.0388, -0.0146] → excludes 0 (significant)
- DistilBERT − RF: -0.0183  95% CI [-0.0308, -0.0050] → excludes 0 (significant)

## Random Forest macro-F1 across 5 seeds (refit)
- seeds [1, 7, 21, 42, 123]: 0.7285, 0.7252, 0.7251, 0.7258, 0.7266
- mean 0.7262, std 0.0012 (XGBoost and SVM are deterministic given fixed data / no subsampling, so their seed variance is negligible; DistilBERT multi-seed variance is left to future work per the compute budget).
