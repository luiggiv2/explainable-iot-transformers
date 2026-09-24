# Revised XAI cohorts

Rows are selected from intersections of correct predictions and aligned one-to-one by `SampleID`. The same rows must be used by both methods in each comparison.

## DistilBERT original vs. XGBoost

| class      |   eligible |   selected |
|:-----------|-----------:|-----------:|
| Benign     |        468 |         60 |
| BruteForce |         93 |         60 |
| DDoS       |       2916 |         60 |
| DoS        |       1168 |         60 |
| Mirai      |       1019 |         60 |
| Recon      |        284 |         60 |
| Spoofing   |        357 |         60 |
| Web        |         67 |         60 |

## DistilBERT original vs. no-`Number`

| class      |   eligible |   selected |
|:-----------|-----------:|-----------:|
| Benign     |        443 |         60 |
| BruteForce |         80 |         60 |
| DDoS       |       2943 |         60 |
| DoS        |       1126 |         60 |
| Mirai      |       1018 |         60 |
| Recon      |        289 |         60 |
| Spoofing   |        354 |         60 |
| Web        |         75 |         60 |

Checkpoint choice used validation performance only. Cohort sampling used true class and joint correctness, not confidence or attribution values.
