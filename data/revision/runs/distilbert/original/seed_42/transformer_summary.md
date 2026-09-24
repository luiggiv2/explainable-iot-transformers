# DistilBERT fine-tuning — CICIoT2023 8-class

Feature set `original`; seed 42; distilbert-base-uncased; max_len 224; batch 32; lr 2e-05; training precision BF16 autocast; validation and test FP32; best epoch 8/15 (validation macro-F1 0.7177); early-stopping patience 4; device mps.

## Per-class metrics
| class      |   precision |   recall |     f1 |   support |
|:-----------|------------:|---------:|-------:|----------:|
| Benign     |      0.7266 |   0.7628 | 0.7443 |       662 |
| BruteForce |      0.728  |   0.4252 | 0.5369 |       214 |
| DDoS       |      0.8291 |   0.8525 | 0.8406 |      3687 |
| DoS        |      0.6786 |   0.6417 | 0.6596 |      1800 |
| Mirai      |      1      |   0.9942 | 0.9971 |      1026 |
| Recon      |      0.6149 |   0.6907 | 0.6506 |       527 |
| Spoofing   |      0.92   |   0.8345 | 0.8751 |       441 |
| Web        |      0.3806 |   0.4372 | 0.4069 |       215 |

## Aggregates and diagnostic cost
| class                               |         f1 |
|:------------------------------------|-----------:|
| _accuracy                           |     0.7863 |
| _macro_avg                          |     0.7139 |
| _weighted_avg                       |     0.7855 |
| _roc_auc_macro_ovr                  |     0.971  |
| _train_time_s                       | 28268.3    |
| _diagnostic_infer_ms_per_sample     |    14.4208 |
| _diagnostic_infer_ms_per_sample_bs1 |    20.2777 |
| _model_size_mb                      |   268.564  |

All reported metrics, the confusion matrix, and downstream statistics must use `test_predictions.parquet` from this same run. Latencies here are diagnostic; the controlled RQ5 benchmark is reported separately.
