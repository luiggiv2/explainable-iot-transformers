# DistilBERT fine-tuning — CICIoT2023 8-class

Feature set `original`; seed 123; distilbert-base-uncased; max_len 224; batch 32; lr 2e-05; training precision BF16 autocast; validation and test FP32; best epoch 11/15 (validation macro-F1 0.7146); early-stopping patience 4; device mps.

## Per-class metrics
| class      |   precision |   recall |     f1 |   support |
|:-----------|------------:|---------:|-------:|----------:|
| Benign     |      0.6922 |   0.7508 | 0.7203 |       662 |
| BruteForce |      0.5231 |   0.528  | 0.5256 |       214 |
| DDoS       |      0.8518 |   0.8123 | 0.8316 |      3687 |
| DoS        |      0.6501 |   0.7111 | 0.6792 |      1800 |
| Mirai      |      0.9951 |   0.9932 | 0.9941 |      1026 |
| Recon      |      0.6564 |   0.6091 | 0.6319 |       527 |
| Spoofing   |      0.9258 |   0.8209 | 0.8702 |       441 |
| Web        |      0.4016 |   0.4651 | 0.431  |       215 |

## Aggregates and diagnostic cost
| class                               |         f1 |
|:------------------------------------|-----------:|
| _accuracy                           |     0.7801 |
| _macro_avg                          |     0.7105 |
| _weighted_avg                       |     0.7825 |
| _roc_auc_macro_ovr                  |     0.9705 |
| _train_time_s                       | 41221.8    |
| _diagnostic_infer_ms_per_sample     |    14.6004 |
| _diagnostic_infer_ms_per_sample_bs1 |    14.3274 |
| _model_size_mb                      |   268.564  |

All reported metrics, the confusion matrix, and downstream statistics must use `test_predictions.parquet` from this same run. Latencies here are diagnostic; the controlled RQ5 benchmark is reported separately.
