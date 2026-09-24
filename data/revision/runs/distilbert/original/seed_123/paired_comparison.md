# Paired model comparison on the leakage-safe test split

All predictions were joined one-to-one by `SampleID` (n=8572). DistilBERT predictions come from the same FP32 run artifact as its reported metrics.

## Macro-F1

| Model | point estimate | bootstrap 95% CI |
|---|---:|---:|
| DistilBERT | 0.7105 | [0.6958, 0.7240] |
| XGBoost | 0.7278 | [0.7128, 0.7420] |
| Random Forest | 0.7223 | [0.7070, 0.7366] |

## Paired macro-F1 differences

| Comparison | point difference | bootstrap 95% CI |
|---|---:|---:|
| DistilBERT - XGBoost | -0.0173 | [-0.0301, -0.0045] |
| DistilBERT - Random Forest | -0.0118 | [-0.0251, +0.0015] |

## Exact McNemar tests

| Comparison | DB only correct | baseline only correct | exact p |
|---|---:|---:|---:|
| DistilBERT vs XGBoost | 315 | 449 | 1.410e-06 |
| DistilBERT vs Random Forest | 548 | 564 | 6.529e-01 |
