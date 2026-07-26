# Explanation robustness (RQ4) — findings

DistilBERT Layer-IG, 60 correctly-classified flows/class, multiplicative Gaussian noise at 1/5/10%, re-attributed toward the true class. Sources: robustness_metrics.csv, robustness_records_partial.csv, 09_ig_noise_floor.py.

## Method validity — IG is deterministic
eps=0 control (identical input attributed twice): Spearman = **1.000** for all classes over both all-39 and top-8 features. Any instability below is genuine input sensitivity, not method noise.

## Metric caveat
Spearman over all 39 features is flat at ~0.47 across noise levels because it is dominated by the ~30 near-zero-attribution features, whose order reshuffles the instant the input moves and then saturates. We therefore report **top-5 feature overlap** and **prediction stability** as the headline metrics.

## Explanation and decision stability track each other

| Class | pred preserved | top-5 Jaccard |
|---|---|---|
| Mirai | 98.3% | 0.448 |
| DDoS | 88.3% | 0.317 |
| DoS | 47.8% | 0.363 |
| Web | 46.1% | 0.267 |
| Spoofing | 33.3% | 0.238 |
| Benign | 30.6% | 0.274 |
| Recon | 5.0% | 0.224 |
| BruteForce | 1.7% | 0.207 |

Cross-class correlation between prediction stability and top-5 explanation stability: **r = +0.854**. Top-5 Jaccard is 0.354 when the prediction is preserved vs 0.244 when it flips.

## Reading
- **Robust where confident:** the well-separated volumetric classes (Mirai ~98% predictions preserved, DDoS ~88%) keep both decision and top features under noise.
- **Fragile where confused:** the low-rate classes the model already confuses (BruteForce ~2%, Recon ~5% predictions preserved) flip class under even 1% noise and their explanations reshuffle — decision and explanation fragility are concentrated in the same classes (ties RQ4 back to RQ1/RQ3).
- Even at best, top-5 stability is moderate (~0.45 for Mirai ≈ 3/5 features retained), so explanations should be read at the class-population level, not trusted per-individual-flow for the low-rate categories.
