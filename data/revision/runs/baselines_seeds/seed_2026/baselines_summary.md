# Classical baselines — CICIoT2023 8-class

Feature set `original`; seed 2026; leakage-safe grouped split; models ['rf', 'xgb'].

## Per-class F1
| class      |     rf |    xgb |
|:-----------|-------:|-------:|
| Benign     | 0.7557 | 0.7516 |
| BruteForce | 0.566  | 0.5521 |
| DDoS       | 0.8224 | 0.8427 |
| DoS        | 0.6378 | 0.69   |
| Mirai      | 0.9976 | 0.9971 |
| Recon      | 0.6854 | 0.6617 |
| Spoofing   | 0.8858 | 0.8865 |
| Web        | 0.4501 | 0.4405 |

## Aggregates and diagnostic cost
| class                           |      rf |    xgb |
|:--------------------------------|--------:|-------:|
| _accuracy                       |  0.7801 | 0.7957 |
| _diagnostic_infer_ms_per_sample |  0.0074 | 0.002  |
| _macro_avg                      |  0.7251 | 0.7278 |
| _model_size_mb                  | 61.5535 | 1.5383 |
| _roc_auc_macro_ovr              |  0.9678 | 0.9733 |
| _train_time_s                   |  1.7736 | 2.2135 |
| _weighted_avg                   |  0.7786 | 0.7959 |

Predictions and probabilities are stored per model and reconstruct every reported confusion matrix. Latencies are diagnostic only.
