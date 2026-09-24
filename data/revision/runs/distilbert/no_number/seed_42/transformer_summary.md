# DistilBERT fine-tuning — CICIoT2023 8-class

Feature set `no_number`; seed 42; distilbert-base-uncased; max_len 224; batch 32; lr 2e-05; training precision BF16 autocast; validation and test FP32; best epoch 8/15 (validation macro-F1 0.7173); early-stopping patience 4; device mps.

## Per-class metrics
| class      |   precision |   recall |     f1 |   support |
|:-----------|------------:|---------:|-------:|----------:|
| Benign     |      0.7203 |   0.7236 | 0.7219 |       662 |
| BruteForce |      0.6692 |   0.4159 | 0.513  |       214 |
| DDoS       |      0.8352 |   0.8424 | 0.8388 |      3687 |
| DoS        |      0.6695 |   0.6572 | 0.6633 |      1800 |
| Mirai      |      0.998  |   0.9951 | 0.9966 |      1026 |
| Recon      |      0.6196 |   0.7078 | 0.6608 |       527 |
| Spoofing   |      0.8886 |   0.8503 | 0.8691 |       441 |
| Web        |      0.3734 |   0.4186 | 0.3947 |       215 |

## Aggregates and diagnostic cost
| class                               |         f1 |
|:------------------------------------|-----------:|
| _accuracy                           |     0.7835 |
| _macro_avg                          |     0.7073 |
| _weighted_avg                       |     0.7831 |
| _roc_auc_macro_ovr                  |     0.9709 |
| _train_time_s                       | 25936.8    |
| _diagnostic_infer_ms_per_sample     |    13.7967 |
| _diagnostic_infer_ms_per_sample_bs1 |    13.4085 |
| _model_size_mb                      |   268.564  |

All reported metrics, the confusion matrix, and downstream statistics must use `test_predictions.parquet` from this same run. Latencies here are diagnostic; the controlled RQ5 benchmark is reported separately.
