# DistilBERT `no_number` feature set — three-seed summary

Seeds 42, 123, and 2026 use an identical leakage-safe split and fixed training protocol. Values below describe training stochasticity; n=3 is too small for strong distributional inference.

## Per-seed results

|   seed |   best_epoch |   validation_macro_f1 |   test_macro_f1 |   test_accuracy |   test_weighted_f1 |   test_roc_auc_macro |   train_time_hours |
|-------:|-------------:|----------------------:|----------------:|----------------:|-------------------:|---------------------:|-------------------:|
|     42 |            8 |                0.7173 |          0.7073 |          0.7835 |             0.7831 |               0.9709 |               7.2  |
|    123 |            7 |                0.7185 |          0.7054 |          0.7807 |             0.7832 |               0.9705 |               8.77 |
|   2026 |           12 |                0.6975 |          0.6878 |          0.7746 |             0.7745 |               0.968  |               8.65 |

## Aggregate across seeds

| metric              |   mean |   sample_sd |    min |    max |
|:--------------------|-------:|------------:|-------:|-------:|
| validation_macro_f1 | 0.7111 |      0.0118 | 0.6975 | 0.7185 |
| test_macro_f1       | 0.7002 |      0.0107 | 0.6878 | 0.7073 |
| test_accuracy       | 0.7796 |      0.0045 | 0.7746 | 0.7835 |
| test_weighted_f1    | 0.7803 |      0.005  | 0.7745 | 0.7832 |
| test_roc_auc_macro  | 0.9698 |      0.0016 | 0.968  | 0.9709 |
| train_time_hours    | 8.211  |      0.8736 | 7.2047 | 8.7746 |

## Difference from fixed classical baselines

| baseline   |   baseline_macro_f1 |   mean_distilbert_minus_baseline |   sample_sd_across_seeds |   min_difference |   max_difference |
|:-----------|--------------------:|---------------------------------:|-------------------------:|-----------------:|-----------------:|
| xgb        |              0.7239 |                          -0.0237 |                   0.0107 |          -0.0361 |          -0.0166 |
| rf         |              0.7225 |                          -0.0223 |                   0.0107 |          -0.0347 |          -0.0152 |

## Per-class F1 across seeds

| class      |   f1_mean |   f1_std |   f1_min |   f1_max |
|:-----------|----------:|---------:|---------:|---------:|
| Benign     |    0.7165 |   0.009  |   0.7062 |   0.7219 |
| BruteForce |    0.5008 |   0.0352 |   0.4612 |   0.5283 |
| DDoS       |    0.8374 |   0.0013 |   0.8362 |   0.8388 |
| DoS        |    0.6667 |   0.0108 |   0.658  |   0.6788 |
| Mirai      |    0.9953 |   0.0012 |   0.9942 |   0.9966 |
| Recon      |    0.6328 |   0.0243 |   0.6164 |   0.6608 |
| Spoofing   |    0.8664 |   0.0042 |   0.8615 |   0.8691 |
| Web        |    0.3855 |   0.0186 |   0.3641 |   0.3977 |

The predeclared median-validation checkpoint for full XAI is **seed 42**. This selection does not inspect test performance.

Test-set bootstrap and across-seed variability quantify different sources of uncertainty and must be reported separately.
