# Explanation robustness under perturbation (RQ4)

DistilBERT Layer-IG; 60 correctly-classified test flows/class; multiplicative Gaussian noise eps in [0.01, 0.05, 0.1] (proto_num fixed, fractions clamped to [0,1]); re-serialized identically and re-attributed toward the true class. Spearman/top-5 Jaccard measure stability of the class attribution map over all perturbed samples; prediction stability is reported separately.

## Overall robustness curve (all classes)

| eps | pred preserved | mean Spearman | mean top-5 Jaccard |
|---|---|---|---|
| 1% | 47.7% | 0.472 | 0.305 |
| 5% | 40.0% | 0.462 | 0.286 |
| 10% | 44.0% | 0.465 | 0.285 |

## Per-class mean Spearman by eps

| Class | 1% | 5% | 10% |
|---|---|---|---|
| Benign | 0.473 | 0.474 | 0.488 |
| BruteForce | 0.428 | 0.444 | 0.440 |
| DDoS | 0.483 | 0.466 | 0.439 |
| DoS | 0.443 | 0.425 | 0.422 |
| Mirai | 0.535 | 0.530 | 0.529 |
| Recon | 0.416 | 0.409 | 0.418 |
| Spoofing | 0.458 | 0.436 | 0.448 |
| Web | 0.536 | 0.515 | 0.533 |

## Per-class prediction preserved by eps

| Class | 1% | 5% | 10% |
|---|---|---|---|
| Benign | 26.7% | 28.3% | 36.7% |
| BruteForce | 1.7% | 0.0% | 3.3% |
| DDoS | 96.7% | 86.7% | 81.7% |
| DoS | 58.3% | 35.0% | 50.0% |
| Mirai | 100.0% | 96.7% | 98.3% |
| Recon | 6.7% | 0.0% | 8.3% |
| Spoofing | 43.3% | 31.7% | 25.0% |
| Web | 48.3% | 41.7% | 48.3% |
