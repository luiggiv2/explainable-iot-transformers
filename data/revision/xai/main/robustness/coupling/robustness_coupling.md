# Prediction-stability / explanation-stability coupling

Source: `robustness_records.parquet` (1440 records, 480 flows, 8 classes).

| Quantity | Value |
|---|---:|
| Pearson r, preserved vs top-5 Jaccard | +0.959 |
| Pearson r, preserved vs top-5 overlap count | +0.954 |
| Exact permutation p (8 classes, Jaccard) | 0.0003 |
| Flow-cluster bootstrap 95% CI (Jaccard, 10,000 resamples) | [+0.897, +0.977] |
| Pearson r without Mirai, DDoS | +0.982 |
| Exact permutation p without Mirai, DDoS | 0.0069 |
| Pooled top-5 Jaccard, preserved / flipped | 0.358 / 0.217 |
| Pooled Spearman, preserved / flipped | 0.533 / 0.435 |

With eight class-level observations the correlation is descriptive.
