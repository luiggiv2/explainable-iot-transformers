# Paired model comparison on the leakage-safe test split

All predictions were joined one-to-one by `SampleID` (n=8572). DistilBERT predictions come from the same FP32 run artifact as its reported metrics.

## Macro-F1

| Model | point estimate | bootstrap 95% CI |
|---|---:|---:|
| DistilBERT | 0.7139 | [0.6990, 0.7275] |
| XGBoost | 0.7278 | [0.7128, 0.7420] |
| Random Forest | 0.7223 | [0.7070, 0.7366] |

## Paired macro-F1 differences

| Comparison | point difference | bootstrap 95% CI |
|---|---:|---:|
| DistilBERT - XGBoost | -0.0139 | [-0.0263, -0.0013] |
| DistilBERT - Random Forest | -0.0084 | [-0.0216, +0.0048] |

## Exact McNemar tests

| Comparison | DB only correct | baseline only correct | exact p |
|---|---:|---:|---:|
| DistilBERT vs XGBoost | 287 | 368 | 1.752e-03 |
| DistilBERT vs Random Forest | 555 | 518 | 2.718e-01 |
