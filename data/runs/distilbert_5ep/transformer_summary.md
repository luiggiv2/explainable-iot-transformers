# DistilBERT fine-tuning — CICIoT2023 8-class (test split, n=9,000)

distilbert-base-uncased, max_len 224, batch 32, lr 2e-05, bf16 autocast, warmup 10%, early stopping on val macro-F1 (patience 2); best epoch 5 (val macro-F1 0.7092); seed 42; device mps.

## Per-class F1
| class      |   precision |   recall |     f1 |   support |
|:-----------|------------:|---------:|-------:|----------:|
| Benign     |      0.6545 |   0.773  | 0.7088 |       696 |
| BruteForce |      0.5525 |   0.4444 | 0.4926 |       225 |
| DDoS       |      0.8386 |   0.8412 | 0.8399 |      3872 |
| DoS        |      0.6723 |   0.6688 | 0.6706 |      1890 |
| Mirai      |      0.9991 |   0.9972 | 0.9981 |      1077 |
| Recon      |      0.6115 |   0.6957 | 0.6508 |       552 |
| Spoofing   |      0.9082 |   0.8103 | 0.8565 |       464 |
| Web        |      0.4741 |   0.2455 | 0.3235 |       224 |

## Aggregates / cost
| class                    |         f1 |
|:-------------------------|-----------:|
| _accuracy                |     0.7831 |
| _macro_avg               |     0.6926 |
| _weighted_avg            |     0.7808 |
| _roc_auc_macro_ovr       |     0.9698 |
| _train_time_s            | 12691.3    |
| _infer_ms_per_sample     |    10.7961 |
| _infer_ms_per_sample_bs1 |    20.5527 |
| _model_size_mb           |   268.564  |

Confusion matrix: confusion_distilbert.csv. Training curve: distilbert_training_log.csv.
