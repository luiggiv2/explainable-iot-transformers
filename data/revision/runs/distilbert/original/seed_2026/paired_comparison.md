# Paired model comparison on the leakage-safe test split

All predictions were joined one-to-one by `SampleID` (n=8572). DistilBERT predictions come from the same FP32 run artifact as its reported metrics.

## Macro-F1

| Model | point estimate | bootstrap 95% CI |
|---|---:|---:|
| DistilBERT | 0.6931 | [0.6786, 0.7073] |
| XGBoost | 0.7278 | [0.7128, 0.7420] |
| Random Forest | 0.7223 | [0.7070, 0.7366] |

## Paired macro-F1 differences

| Comparison | point difference | bootstrap 95% CI |
|---|---:|---:|
| DistilBERT - XGBoost | -0.0347 | [-0.0481, -0.0211] |
| DistilBERT - Random Forest | -0.0292 | [-0.0428, -0.0148] |

## Exact McNemar tests

| Comparison | DB only correct | baseline only correct | exact p |
|---|---:|---:|---:|
| DistilBERT vs XGBoost | 305 | 435 | 1.997e-06 |
| DistilBERT vs Random Forest | 555 | 567 | 7.426e-01 |
