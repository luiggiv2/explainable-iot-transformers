# DistilBERT original vs. no-`Number` comparison

All three pairs use the same leakage-safe test rows and are aligned one-to-one by `SampleID` (n=8572 per seed). Every run passed its saved-prediction and confusion-matrix consistency checks.

## Results by training seed

|   seed |   original_test_macro_f1 |   no_number_test_macro_f1 |   test_macro_f1_delta |   original_test_accuracy |   no_number_test_accuracy |   test_accuracy_delta |
|-------:|-------------------------:|--------------------------:|----------------------:|-------------------------:|--------------------------:|----------------------:|
|     42 |                   0.7139 |                    0.7073 |               -0.0066 |                   0.7863 |                    0.7835 |               -0.0028 |
|    123 |                   0.7105 |                    0.7054 |               -0.0051 |                   0.7801 |                    0.7807 |                0.0006 |
|   2026 |                   0.6931 |                    0.6878 |               -0.0053 |                   0.7806 |                    0.7746 |               -0.0059 |

Across seeds, removing `Number` changed test macro-F1 by **-0.0057 ± 0.0008** (mean ± sample SD). The three differences have the same sign, but three seeds are not enough for strong distributional claims about training variability.

## Paired test-row bootstrap

|   seed |   delta |   delta_ci_lower |   delta_ci_upper |
|-------:|--------:|-----------------:|-----------------:|
|     42 | -0.0066 |          -0.0176 |           0.0048 |
|    123 | -0.0051 |          -0.0161 |           0.0057 |
|   2026 | -0.0053 |          -0.0171 |           0.0062 |

Each interval uses 4,000 paired resamples. These intervals describe test-row sampling uncertainty conditional on each fitted pair; they do not replace the across-seed summary above.

## Exact McNemar tests for accuracy

|   seed |   original_only_correct |   no_number_only_correct |   discordant |   exact_p |
|-------:|------------------------:|-------------------------:|-------------:|----------:|
|     42 |                     253 |                      229 |          482 |   0.2948  |
|    123 |                     328 |                      333 |          661 |   0.8764  |
|   2026 |                     321 |                      270 |          591 |   0.03962 |

McNemar's test concerns paired correctness, not macro-F1. It should not be used to support a claim about the macro-F1 difference.

## Mean per-class F1 change across seeds

| class      |   original_f1_mean |   no_number_f1_mean |   delta_mean |
|:-----------|-------------------:|--------------------:|-------------:|
| Benign     |             0.7245 |              0.7165 |      -0.0081 |
| BruteForce |             0.5123 |              0.5008 |      -0.0115 |
| DDoS       |             0.8377 |              0.8374 |      -0.0003 |
| DoS        |             0.6716 |              0.6667 |      -0.0049 |
| Mirai      |             0.9959 |              0.9953 |      -0.0006 |
| Recon      |             0.6285 |              0.6328 |       0.0043 |
| Spoofing   |             0.8689 |              0.8664 |      -0.0026 |
| Web        |             0.4071 |              0.3855 |      -0.0216 |

## Interpretation

The point estimates show a small, consistent reduction in overall macro-F1 rather than a collapse, although every per-seed bootstrap interval includes zero. The largest mean change is in Web, while Recon improves slightly. A restrained reading is therefore appropriate: the result is consistent with a modest contribution from the window-size field, and the transformer retains most of its within-dataset performance without it. The experiment does not establish transfer to another dataset.

The validation-median checkpoints selected without test inspection are seed **123** for `original` and seed **42** for `no_number`.
