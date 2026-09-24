# DistilBERT `original` feature set — three-seed summary

Seeds 42, 123, and 2026 use an identical leakage-safe split and fixed training protocol. Values below describe training stochasticity; n=3 is too small for strong distributional inference.

## Per-seed results

|   seed |   best_epoch |   validation_macro_f1 |   test_macro_f1 |   test_accuracy |   test_weighted_f1 |   test_roc_auc_macro |   train_time_hours |
|-------:|-------------:|----------------------:|----------------:|----------------:|-------------------:|---------------------:|-------------------:|
|     42 |            8 |                0.7177 |          0.7139 |          0.7863 |             0.7855 |               0.971  |               7.85 |
|    123 |           11 |                0.7146 |          0.7105 |          0.7801 |             0.7825 |               0.9705 |              11.45 |
|   2026 |           10 |                0.6978 |          0.6931 |          0.7806 |             0.7805 |               0.9687 |               8.8  |

## Aggregate across seeds

| metric              |   mean |   sample_sd |    min |     max |
|:--------------------|-------:|------------:|-------:|--------:|
| validation_macro_f1 | 0.71   |      0.0107 | 0.6978 |  0.7177 |
| test_macro_f1       | 0.7058 |      0.0112 | 0.6931 |  0.7139 |
| test_accuracy       | 0.7823 |      0.0034 | 0.7801 |  0.7863 |
| test_weighted_f1    | 0.7828 |      0.0025 | 0.7805 |  0.7855 |
| test_roc_auc_macro  | 0.9701 |      0.0012 | 0.9687 |  0.971  |
| train_time_hours    | 9.3683 |      1.8647 | 7.8523 | 11.4505 |

## Difference from fixed classical baselines

| baseline   |   baseline_macro_f1 |   mean_distilbert_minus_baseline |   sample_sd_across_seeds |   min_difference |   max_difference |
|:-----------|--------------------:|---------------------------------:|-------------------------:|-----------------:|-----------------:|
| xgb        |              0.7278 |                          -0.0219 |                   0.0112 |          -0.0347 |          -0.0139 |
| rf         |              0.7223 |                          -0.0164 |                   0.0112 |          -0.0292 |          -0.0084 |

## Per-class F1 across seeds

| class      |   f1_mean |   f1_std |   f1_min |   f1_max |
|:-----------|----------:|---------:|---------:|---------:|
| Benign     |    0.7245 |   0.018  |   0.7091 |   0.7443 |
| BruteForce |    0.5123 |   0.0333 |   0.4744 |   0.5369 |
| DDoS       |    0.8377 |   0.0053 |   0.8316 |   0.8408 |
| DoS        |    0.6716 |   0.0105 |   0.6596 |   0.6792 |
| Mirai      |    0.9959 |   0.0016 |   0.9941 |   0.9971 |
| Recon      |    0.6285 |   0.0239 |   0.6031 |   0.6506 |
| Spoofing   |    0.8689 |   0.0069 |   0.8615 |   0.8751 |
| Web        |    0.4071 |   0.0238 |   0.3834 |   0.431  |

The predeclared median-validation checkpoint for full XAI is **seed 123**. This selection does not inspect test performance.

Test-set bootstrap and across-seed variability quantify different sources of uncertainty and must be reported separately.
