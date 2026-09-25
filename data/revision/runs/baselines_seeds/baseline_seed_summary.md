# Seed variance: tree baselines vs DistilBERT

Random Forest and XGBoost re-fitted with `scripts/03_baselines.py` (fixed configuration, seeds 42/123/2026, same splits). DistilBERT values are the three fine-tuning runs (`results/distilbert_original_seeds.csv`).

| Model | Seed | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| distilbert | 42 | 0.7139 | 0.7863 |
| distilbert | 123 | 0.7105 | 0.7801 |
| distilbert | 2026 | 0.6931 | 0.7806 |
| rf | 42 | 0.7223 | 0.7820 |
| rf | 123 | 0.7242 | 0.7817 |
| rf | 2026 | 0.7251 | 0.7801 |
| xgb | 42 | 0.7278 | 0.7957 |
| xgb | 123 | 0.7278 | 0.7957 |
| xgb | 2026 | 0.7278 | 0.7957 |

| Model | macro-F1 mean ± SD | range | accuracy mean ± SD |
|---|---:|---|---:|
| distilbert | 0.7058 ± 0.0112 | [0.6931, 0.7139] | 0.7823 ± 0.0034 |
| rf | 0.7238 ± 0.0014 | [0.7223, 0.7251] | 0.7813 ± 0.0010 |
| xgb | 0.7278 ± 0.0000 | [0.7278, 0.7278] | 0.7957 ± 0.0000 |

Seed 42 reproduces the reported baseline run exactly: RF True, XGBoost True.

## XGBoost early stopping (validation mlogloss)

| Run | best iteration (0-based) | best validation mlogloss | rounds fitted |
|---|---:|---:|---:|
| seed_42 | 127 | 0.4215 | 158 |
| seed_123 | 127 | 0.4215 | 158 |
| seed_2026 | 127 | 0.4215 | 158 |
| original_run | 127 | 0.4215 | 158 |

XGBoost uses no row or column subsampling in this configuration, so its seed has no effect on the fitted model when the outputs are identical across seeds.
