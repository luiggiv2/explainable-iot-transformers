# Classical baselines — CICIoT2023 8-class

Feature set `no_number_protocol`; seed 42; leakage-safe grouped split; models ['rf', 'xgb'].

## Per-class F1
| class      |     rf |    xgb |
|:-----------|-------:|-------:|
| Benign     | 0.7554 | 0.7495 |
| BruteForce | 0.5515 | 0.5393 |
| DDoS       | 0.8228 | 0.8411 |
| DoS        | 0.638  | 0.6844 |
| Mirai      | 0.9976 | 0.9976 |
| Recon      | 0.6876 | 0.6508 |
| Spoofing   | 0.8865 | 0.8902 |
| Web        | 0.4463 | 0.4136 |

## Aggregates and diagnostic cost
| class                           |      rf |    xgb |
|:--------------------------------|--------:|-------:|
| _accuracy                       |  0.7801 | 0.7927 |
| _diagnostic_infer_ms_per_sample |  0.0173 | 0.0027 |
| _macro_avg                      |  0.7232 | 0.7208 |
| _model_size_mb                  | 61.9491 | 1.9491 |
| _roc_auc_macro_ovr              |  0.9679 | 0.9729 |
| _train_time_s                   |  5.5505 | 5.818  |
| _weighted_avg                   |  0.7785 | 0.7924 |

Predictions and probabilities are stored per model and reconstruct every reported confusion matrix. Latencies are diagnostic only.
