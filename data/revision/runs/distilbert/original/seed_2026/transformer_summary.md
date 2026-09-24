# DistilBERT fine-tuning — CICIoT2023 8-class

Feature set `original`; seed 2026; distilbert-base-uncased; max_len 224; batch 32; lr 2e-05; training precision BF16 autocast; validation and test FP32; best epoch 10/15 (validation macro-F1 0.6978); early-stopping patience 4; device mps.

## Per-class metrics
| class      |   precision |   recall |     f1 |   support |
|:-----------|------------:|---------:|-------:|----------:|
| Benign     |      0.6535 |   0.7749 | 0.7091 |       662 |
| BruteForce |      0.4722 |   0.4766 | 0.4744 |       214 |
| DDoS       |      0.8428 |   0.8389 | 0.8408 |      3687 |
| DoS        |      0.6723 |   0.6794 | 0.6759 |      1800 |
| Mirai      |      0.9971 |   0.9961 | 0.9966 |      1026 |
| Recon      |      0.6729 |   0.5465 | 0.6031 |       527 |
| Spoofing   |      0.8929 |   0.8322 | 0.8615 |       441 |
| Web        |      0.3807 |   0.386  | 0.3834 |       215 |

## Aggregates and diagnostic cost
| class                               |         f1 |
|:------------------------------------|-----------:|
| _accuracy                           |     0.7806 |
| _macro_avg                          |     0.6931 |
| _weighted_avg                       |     0.7805 |
| _roc_auc_macro_ovr                  |     0.9687 |
| _train_time_s                       | 31687.9    |
| _diagnostic_infer_ms_per_sample     |    12.8794 |
| _diagnostic_infer_ms_per_sample_bs1 |    13.5141 |
| _model_size_mb                      |   268.564  |

All reported metrics, the confusion matrix, and downstream statistics must use `test_predictions.parquet` from this same run. Latencies here are diagnostic; the controlled RQ5 benchmark is reported separately.
