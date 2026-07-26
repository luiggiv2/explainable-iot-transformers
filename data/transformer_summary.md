# DistilBERT fine-tuning — CICIoT2023 8-class (test split, n=9,000)

distilbert-base-uncased, max_len 224, batch 32, lr 2e-05, bf16 autocast, warmup 10%, early stopping on val macro-F1 (patience 3); best epoch 10 (val macro-F1 0.7247); seed 42; device mps.

## Per-class F1
| class      |   precision |   recall |     f1 |   support |
|:-----------|------------:|---------:|-------:|----------:|
| Benign     |      0.6784 |   0.773  | 0.7226 |       696 |
| BruteForce |      0.5756 |   0.5244 | 0.5488 |       225 |
| DDoS       |      0.8507 |   0.8182 | 0.8341 |      3872 |
| DoS        |      0.6546 |   0.7058 | 0.6792 |      1890 |
| Mirai      |      0.9972 |   0.9972 | 0.9972 |      1077 |
| Recon      |      0.6437 |   0.5761 | 0.608  |       552 |
| Spoofing   |      0.8961 |   0.8362 | 0.8651 |       464 |
| Web        |      0.3898 |   0.4107 | 0.4    |       224 |

## Aggregates / cost
| class                    |         f1 |
|:-------------------------|-----------:|
| _accuracy                |     0.7811 |
| _macro_avg               |     0.7069 |
| _weighted_avg            |     0.7823 |
| _roc_auc_macro_ovr       |     0.9706 |
| _train_time_s            | 21977.1    |
| _infer_ms_per_sample     |    10.3697 |
| _infer_ms_per_sample_bs1 |    15.5238 |
| _model_size_mb           |   268.564  |

Confusion matrix: confusion_distilbert.csv. Training curve: distilbert_training_log.csv.
