# Classical baselines — CICIoT2023 8-class

Feature set `no_number`; seed 42; leakage-safe grouped split; models ['rf', 'xgb'].

## Per-class F1
| class      |     rf |    xgb |
|:-----------|-------:|-------:|
| Benign     | 0.7568 | 0.7528 |
| BruteForce | 0.5589 | 0.5526 |
| DDoS       | 0.8227 | 0.8406 |
| DoS        | 0.6376 | 0.6858 |
| Mirai      | 0.9976 | 0.9976 |
| Recon      | 0.6919 | 0.6532 |
| Spoofing   | 0.8871 | 0.8844 |
| Web        | 0.4274 | 0.4241 |

## Aggregates and diagnostic cost
| class                           |      rf |    xgb |
|:--------------------------------|--------:|-------:|
| _accuracy                       |  0.7802 | 0.7933 |
| _diagnostic_infer_ms_per_sample |  0.0198 | 0.0032 |
| _macro_avg                      |  0.7225 | 0.7239 |
| _model_size_mb                  | 62.0861 | 1.9994 |
| _roc_auc_macro_ovr              |  0.9677 | 0.9729 |
| _train_time_s                   |  5.5391 | 5.9748 |
| _weighted_avg                   |  0.7784 | 0.7932 |

Predictions and probabilities are stored per model and reconstruct every reported confusion matrix. Latencies are diagnostic only.
