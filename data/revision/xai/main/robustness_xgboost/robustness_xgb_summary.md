# Explanation robustness, XGBoost/TreeSHAP arm

Same 480-row cohort and the same deterministic perturbation as scripts/28 (seed 42; count fields rounded; proto_num fixed; fractions clipped to [0, 1]). TreeSHAP toward the true class, |values| L1-normalized per sample.

XGBoost classifies 431 of 480 cohort rows correctly before perturbation; prediction preservation and attribution metrics below are over those rows.

## Overall

| Perturbation | n | XGB prediction preserved | Mean Spearman | Mean top-5 overlap | DistilBERT preserved (same cohort) | DistilBERT top-5 overlap | `num` unchanged after rounding |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 431 | 83.3% | 0.982 | 4.47/5 | 56.7% | 2.27/5 | 74.6% |
| 5% | 431 | 74.7% | 0.963 | 4.02/5 | 49.4% | 2.03/5 | 46.9% |
| 10% | 431 | 66.8% | 0.947 | 3.79/5 | 43.3% | 1.96/5 | 25.4% |

## By class, pooled across perturbation levels

| Class | n rows | XGB preserved | XGB top-5 Jaccard | XGB Spearman | DistilBERT preserved | DistilBERT top-5 Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| Benign | 55 | 64.2% | 0.736 | 0.971 | 20.0% | 0.220 |
| BruteForce | 48 | 63.9% | 0.693 | 0.952 | 8.3% | 0.160 |
| DDoS | 59 | 91.5% | 0.733 | 0.949 | 68.9% | 0.337 |
| DoS | 57 | 80.1% | 0.569 | 0.946 | 76.1% | 0.357 |
| Mirai | 60 | 100.0% | 0.926 | 0.992 | 100.0% | 0.368 |
| Recon | 54 | 53.7% | 0.747 | 0.968 | 21.7% | 0.230 |
| Spoofing | 59 | 75.7% | 0.736 | 0.974 | 53.9% | 0.313 |
| Web | 39 | 60.7% | 0.652 | 0.957 | 49.4% | 0.310 |

XGBoost class-level Pearson r (prediction preserved vs top-5 Jaccard) = +0.454 (exact permutation p = 0.2632; 8 classes, descriptive).

The perturbation is benign multiplicative jitter, not an adversarial attack.
