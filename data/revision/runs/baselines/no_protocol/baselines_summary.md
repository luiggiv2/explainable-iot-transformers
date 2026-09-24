# Classical baselines — CICIoT2023 8-class

Feature set `no_protocol`; seed 42; leakage-safe grouped split; models ['rf', 'xgb'].

## Per-class F1
| class      |     rf |    xgb |
|:-----------|-------:|-------:|
| Benign     | 0.7512 | 0.7537 |
| BruteForce | 0.558  | 0.5585 |
| DDoS       | 0.8243 | 0.8438 |
| DoS        | 0.6427 | 0.6906 |
| Mirai      | 0.9976 | 0.9976 |
| Recon      | 0.6892 | 0.6599 |
| Spoofing   | 0.8857 | 0.8905 |
| Web        | 0.44   | 0.4354 |

## Aggregates and diagnostic cost
| class                           |      rf |    xgb |
|:--------------------------------|--------:|-------:|
| _accuracy                       |  0.7814 | 0.7968 |
| _diagnostic_infer_ms_per_sample |  0.0217 | 0.0022 |
| _macro_avg                      |  0.7236 | 0.7287 |
| _model_size_mb                  | 61.2752 | 1.6939 |
| _roc_auc_macro_ovr              |  0.9682 | 0.9734 |
| _train_time_s                   |  5.5205 | 5.6433 |
| _weighted_avg                   |  0.7798 | 0.7968 |

Predictions and probabilities are stored per model and reconstruct every reported confusion matrix. Latencies are diagnostic only.
