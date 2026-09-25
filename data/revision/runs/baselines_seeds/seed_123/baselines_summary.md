# Classical baselines — CICIoT2023 8-class

Feature set `original`; seed 123; leakage-safe grouped split; models ['rf', 'xgb'].

## Per-class F1
| class      |     rf |    xgb |
|:-----------|-------:|-------:|
| Benign     | 0.7503 | 0.7516 |
| BruteForce | 0.562  | 0.5521 |
| DDoS       | 0.8248 | 0.8427 |
| DoS        | 0.6442 | 0.69   |
| Mirai      | 0.9976 | 0.9971 |
| Recon      | 0.6864 | 0.6617 |
| Spoofing   | 0.8873 | 0.8865 |
| Web        | 0.4407 | 0.4405 |

## Aggregates and diagnostic cost
| class                           |      rf |    xgb |
|:--------------------------------|--------:|-------:|
| _accuracy                       |  0.7817 | 0.7957 |
| _diagnostic_infer_ms_per_sample |  0.0075 | 0.0016 |
| _macro_avg                      |  0.7242 | 0.7278 |
| _model_size_mb                  | 61.4234 | 1.5382 |
| _roc_auc_macro_ovr              |  0.968  | 0.9733 |
| _train_time_s                   |  1.7534 | 2.2807 |
| _weighted_avg                   |  0.7803 | 0.7959 |

Predictions and probabilities are stored per model and reconstruct every reported confusion matrix. Latencies are diagnostic only.
