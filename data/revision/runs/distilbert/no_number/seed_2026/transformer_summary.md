# DistilBERT fine-tuning — CICIoT2023 8-class

Feature set `no_number`; seed 2026; distilbert-base-uncased; max_len 224; batch 32; lr 2e-05; training precision BF16 autocast; validation and test FP32; best epoch 12/15 (validation macro-F1 0.6975); early-stopping patience 4; device mps.

## Per-class metrics
| class      |   precision |   recall |     f1 |   support |
|:-----------|------------:|---------:|-------:|----------:|
| Benign     |      0.7099 |   0.7024 | 0.7062 |       662 |
| BruteForce |      0.4645 |   0.4579 | 0.4612 |       214 |
| DDoS       |      0.8306 |   0.8419 | 0.8362 |      3687 |
| DoS        |      0.6674 |   0.6489 | 0.658  |      1800 |
| Mirai      |      0.9932 |   0.9951 | 0.9942 |      1026 |
| Recon      |      0.6025 |   0.6414 | 0.6213 |       527 |
| Spoofing   |      0.8929 |   0.8322 | 0.8615 |       441 |
| Web        |      0.3607 |   0.3674 | 0.3641 |       215 |

## Aggregates and diagnostic cost
| class                               |         f1 |
|:------------------------------------|-----------:|
| _accuracy                           |     0.7746 |
| _macro_avg                          |     0.6878 |
| _weighted_avg                       |     0.7745 |
| _roc_auc_macro_ovr                  |     0.968  |
| _train_time_s                       | 31153.3    |
| _diagnostic_infer_ms_per_sample     |    13.5729 |
| _diagnostic_infer_ms_per_sample_bs1 |    14.4107 |
| _model_size_mb                      |   268.564  |

All reported metrics, the confusion matrix, and downstream statistics must use `test_predictions.parquet` from this same run. Latencies here are diagnostic; the controlled RQ5 benchmark is reported separately.
