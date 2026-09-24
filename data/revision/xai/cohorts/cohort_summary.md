# Revised XAI cohorts

Rows are selected from intersections of correct predictions and aligned one-to-one by `SampleID`. The same rows must be used by both methods in each comparison.

## DistilBERT original vs. XGBoost

| class      |   eligible |   selected |
|:-----------|-----------:|-----------:|
| Benign     |        468 |        120 |
| BruteForce |         93 |         93 |
| DDoS       |       2916 |        120 |
| DoS        |       1168 |        120 |
| Mirai      |       1019 |        120 |
| Recon      |        284 |        120 |
| Spoofing   |        357 |        120 |
| Web        |         67 |         67 |

## DistilBERT original vs. no-`Number`

| class      |   eligible |   selected |
|:-----------|-----------:|-----------:|
| Benign     |        443 |        120 |
| BruteForce |         80 |         80 |
| DDoS       |       2943 |        120 |
| DoS        |       1126 |        120 |
| Mirai      |       1018 |        120 |
| Recon      |        289 |        120 |
| Spoofing   |        354 |        120 |
| Web        |         75 |         75 |

Checkpoint choice used validation performance only. Cohort sampling used true class and joint correctness, not confidence or attribution values.
