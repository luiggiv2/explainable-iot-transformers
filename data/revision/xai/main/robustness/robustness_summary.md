# Revised explanation robustness analysis

The analysis uses 480 correctly classified test flows (60 per class where available), with deterministic multiplicative perturbations of 1%, 5%, and 10%. Protocol identity and zero-valued indicator support remain unchanged.

## Overall results

| Perturbation | Prediction preserved | Mean Spearman | Mean top-5 overlap |
|---:|---:|---:|---:|
| 1% | 56.7% | 0.521 | 2.27/5 |
| 5% | 49.4% | 0.476 | 2.03/5 |
| 10% | 43.3% | 0.454 | 1.96/5 |

## Results by class, pooled across perturbation levels

| Class | Prediction preserved | Mean Spearman | Mean top-5 overlap |
|---|---:|---:|---:|
| Benign | 20.0% | 0.435 | 1.69/5 |
| BruteForce | 8.3% | 0.426 | 1.26/5 |
| DDoS | 68.9% | 0.425 | 2.37/5 |
| DoS | 76.1% | 0.549 | 2.51/5 |
| Mirai | 100.0% | 0.551 | 2.62/5 |
| Recon | 21.7% | 0.447 | 1.74/5 |
| Spoofing | 53.9% | 0.503 | 2.26/5 |
| Web | 49.4% | 0.534 | 2.26/5 |

Across the eight classes, prediction preservation and mean top-5 overlap have Pearson r = +0.954 (two-sided exact permutation p = 0.0002). With only eight class-level observations, this association is descriptive and should not be read as a broadly generalizable estimate.

The perturbations probe local sensitivity within this representation. They are not adversarial attacks and do not establish robustness under distribution shift.
