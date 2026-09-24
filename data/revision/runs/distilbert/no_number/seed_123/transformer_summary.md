# DistilBERT fine-tuning — CICIoT2023 8-class

Feature set `no_number`; seed 123; distilbert-base-uncased; max_len 224; batch 32; lr 2e-05; training precision BF16 autocast; validation and test FP32; best epoch 7/15 (validation macro-F1 0.7185); early-stopping patience 4; device mps.

## Per-class metrics
| class      |   precision |   recall |     f1 |   support |
|:-----------|------------:|---------:|-------:|----------:|
| Benign     |      0.6891 |   0.7568 | 0.7214 |       662 |
| BruteForce |      0.5333 |   0.5234 | 0.5283 |       214 |
| DDoS       |      0.8472 |   0.8272 | 0.8371 |      3687 |
| DoS        |      0.6638 |   0.6944 | 0.6788 |      1800 |
| Mirai      |      0.9961 |   0.9942 | 0.9951 |      1026 |
| Recon      |      0.6885 |   0.5579 | 0.6164 |       527 |
| Spoofing   |      0.9278 |   0.8163 | 0.8685 |       441 |
| Web        |      0.3355 |   0.4884 | 0.3977 |       215 |

## Aggregates and diagnostic cost
| class                               |         f1 |
|:------------------------------------|-----------:|
| _accuracy                           |     0.7807 |
| _macro_avg                          |     0.7054 |
| _weighted_avg                       |     0.7832 |
| _roc_auc_macro_ovr                  |     0.9705 |
| _train_time_s                       | 31588.7    |
| _diagnostic_infer_ms_per_sample     |    13.4951 |
| _diagnostic_infer_ms_per_sample_bs1 |    13.428  |
| _model_size_mb                      |   268.564  |

All reported metrics, the confusion matrix, and downstream statistics must use `test_predictions.parquet` from this same run. Latencies here are diagnostic; the controlled RQ5 benchmark is reported separately.
