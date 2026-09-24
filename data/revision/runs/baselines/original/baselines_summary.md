# Classical baselines — CICIoT2023 8-class

Feature set `original`; seed 42; leakage-safe grouped split; models ['rf', 'xgb', 'svm'].

## Per-class F1
| class      |     rf |    svm |    xgb |
|:-----------|-------:|-------:|-------:|
| Benign     | 0.7557 | 0.7215 | 0.7516 |
| BruteForce | 0.5604 | 0.4691 | 0.5521 |
| DDoS       | 0.8251 | 0.8309 | 0.8427 |
| DoS        | 0.6434 | 0.6524 | 0.69   |
| Mirai      | 0.998  | 0.9956 | 0.9971 |
| Recon      | 0.6834 | 0.6486 | 0.6617 |
| Spoofing   | 0.8889 | 0.7143 | 0.8865 |
| Web        | 0.4232 | 0.3057 | 0.4405 |

## Aggregates and diagnostic cost
| class                           |      rf |     svm |    xgb |
|:--------------------------------|--------:|--------:|-------:|
| _accuracy                       |  0.782  |  0.7695 | 0.7957 |
| _diagnostic_infer_ms_per_sample |  0.0073 |  0.8323 | 0.002  |
| _macro_avg                      |  0.7223 |  0.6672 | 0.7278 |
| _model_size_mb                  | 61.5075 |  0.9934 | 1.5383 |
| _roc_auc_macro_ovr              |  0.9677 |  0.9631 | 0.9733 |
| _train_time_s                   |  1.684  | 65.333  | 2.2064 |
| _weighted_avg                   |  0.7802 |  0.7653 | 0.7959 |

Predictions and probabilities are stored per model and reconstruct every reported confusion matrix. Latencies are diagnostic only.
