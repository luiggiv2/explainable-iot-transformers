# Classical baselines — CICIoT2023 8-class (test split, n=9,000)

Subset per data/subset_composition.md; seed 42. RF: 300 trees. XGBoost: hist, lr 0.1, depth 8, early stopping on val. SVM: RBF, C=10.0 selected on val, standardized inputs.

## Per-class F1
| class      |     rf |    svm |    xgb |
|:-----------|-------:|-------:|-------:|
| Benign     | 0.7368 | 0.6877 | 0.7468 |
| BruteForce | 0.5616 | 0.4717 | 0.5532 |
| DDoS       | 0.8299 | 0.8372 | 0.8439 |
| DoS        | 0.6562 | 0.6654 | 0.6955 |
| Mirai      | 0.9981 | 0.9963 | 0.9986 |
| Recon      | 0.6747 | 0.6384 | 0.6751 |
| Spoofing   | 0.8795 | 0.6697 | 0.8776 |
| Web        | 0.4697 | 0.2763 | 0.4845 |

## Aggregates / cost
| class                |      rf |     svm |    xgb |
|:---------------------|--------:|--------:|-------:|
| _accuracy            |  0.7849 |  0.7694 | 0.7993 |
| _infer_ms_per_sample |  0.0086 |  0.8131 | 0.0016 |
| _macro_avg           |  0.7258 |  0.6554 | 0.7344 |
| _model_size_mb       | 59.9703 |  0.9727 | 1.5599 |
| _roc_auc_macro_ovr   |  0.9695 |  0.9629 | 0.9743 |
| _train_time_s        |  1.7235 | 63.7777 | 2.1748 |
| _weighted_avg        |  0.7837 |  0.7647 | 0.7989 |

Confusion matrices: confusion_rf.csv, confusion_xgb.csv, confusion_svm.csv
