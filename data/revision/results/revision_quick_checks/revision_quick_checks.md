# Revision quick checks

## 1. SSH in BruteForce windows

- test: 31.8% of 214 BruteForce windows have SSH > 0 (other categories 0.6%; Benign 2.6%); 58.1% of SSH-positive test windows are BruteForce.
- train: 31.7% of 1071 (other 0.5%).
- XAI cohort: 67.7% of 93 BruteForce rows.

## 2. Recon composition (test)

- VULNERABILITYSCAN: 285 (54.1%)
- RECON-HOSTDISCOVERY: 102 (19.4%)
- RECON-OSSCAN: 75 (14.2%)
- RECON-PORTSCAN: 63 (12.0%)
- RECON-PINGSWEEP: 2 (0.4%)

## 3. Batch-1 / batch-64 per-sample latency ratio

| Model | model_only | with_input_preparation |
|---|---:|---:|
| distilbert | 1.08x | 1.11x |
| rf | 31.42x | 31.31x |
| svm | 1.15x | 1.36x |
| xgb | 18.83x | 30.81x |

## 4. Layer-IG |convergence delta| (logit units)

n = 880; mean 0.0487; median 0.0249; p95 0.1838; max 1.1857.

## 5. McNemar, no_number vs original

- seed 42: p = 0.2948
- seed 123: p = 0.8764
- seed 2026: p = 0.0396

## 6. Concentration of the no_number per-class F1 loss

Summed negative mean per-class deltas -0.0495; Web + BruteForce share 66.8%.

## 7. Exact-feature duplicate groups with conflicting categories (all 59,997 rows)

- duplicate groups 1690 (3658 rows); conflicting-category groups 496 (1125 rows).
- category combinations: DDoS / DoS: 484; Benign / Spoofing: 4; Spoofing / Web: 4; Recon / Spoofing: 1; BruteForce / Spoofing: 1; Benign / Recon: 1; Benign / Web: 1.
- rows in conflicting groups by category: DDoS: 553; DoS: 533; Spoofing: 22; Web: 7; Benign: 7; Recon: 2; BruteForce: 1.

## 7b. Capture-file overlap

100.0% of test rows come from raw CSV files that also contribute train rows (63 files in test, 63 in train); the split is grouped by serialized content, not by capture file or time.

## 8. WordPiece token length (incl. [CLS]/[SEP])

| Split | n | median | p99 | max | > 224 |
|---|---:|---:|---:|---:|---:|
| train | 42853 | 181 | 212 | 226 | 2 |
| val | 8572 | 180 | 211 | 224 | 0 |
| test | 8572 | 180 | 211 | 222 | 0 |

## 9. Paired bootstrap re-derived (4,000 resamples, RNG seed 42)

| Seed | Comparison | 95% CI | Bonferroni 99.17% CI | McNemar p | x6 |
|---:|---|---|---|---:|---:|
| 42 | DistilBERT - XGBoost | [-0.0263, -0.0013] | [-0.0317, +0.0025] | 1.75e-03 | 1.05e-02 |
| 42 | DistilBERT - RF | [-0.0216, +0.0048] | [-0.0264, +0.0094] | 2.72e-01 | 1.00e+00 |
| 123 | DistilBERT - XGBoost | [-0.0301, -0.0045] | [-0.0341, -0.0008] | 1.41e-06 | 8.46e-06 |
| 123 | DistilBERT - RF | [-0.0251, +0.0015] | [-0.0298, +0.0064] | 6.53e-01 | 1.00e+00 |
| 2026 | DistilBERT - XGBoost | [-0.0481, -0.0211] | [-0.0523, -0.0162] | 2.00e-06 | 1.20e-05 |
| 2026 | DistilBERT - RF | [-0.0428, -0.0148] | [-0.0475, -0.0097] | 7.43e-01 | 1.00e+00 |
