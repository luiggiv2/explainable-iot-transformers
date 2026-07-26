# Statistical hardening (reviewer Phase B)

## McNemar exact test (paired, shared n=9000 test rows)
- accuracy: DistilBERT 0.7811, XGBoost 0.7993, RF 0.7849
- DistilBERT vs XGBoost: discordant b(DB right, XGBoost wrong)=328, c(DB wrong, XGBoost right)=492; exact McNemar p=1.126e-08 → significant at alpha=0.05.
- DistilBERT vs RF: discordant b(DB right, RF wrong)=594, c(DB wrong, RF right)=628; exact McNemar p=3.452e-01 → NOT significant at alpha=0.05.

## RQ4 coupling (prediction-preserved vs top-5 Jaccard, n=8 classes)
- Pearson r = +0.854; sample-level bootstrap 95% CI [+0.774, +0.906] (3000 resamples)
- permutation p = 0.0070 (20000 permutations)
- leave-two-out (drop Mirai, DDoS): r = +0.782, p = 0.066 on 6 classes

## RQ4 prediction-preservation across noise levels
| Class | 1% | 5% | 10% | monotone non-increasing? |
|---|---|---|---|---|
| Benign | 26.7% | 28.3% | 36.7% | **NO** |
| BruteForce | 1.7% | 0.0% | 3.3% | **NO** |
| DDoS | 96.7% | 86.7% | 81.7% | yes |
| DoS | 58.3% | 35.0% | 50.0% | **NO** |
| Mirai | 100.0% | 96.7% | 98.3% | **NO** |
| Recon | 6.7% | 0.0% | 8.3% | **NO** |
| Spoofing | 43.3% | 31.7% | 25.0% | yes |
| Web | 48.3% | 41.7% | 48.3% | **NO** |

6/8 classes are non-monotone: prediction preservation does not fall monotonically as noise grows, indicating the levels do not behave as a graded robustness sweep (small per-class n).

## RQ2/RQ3 IG ranking stability (1000-sample bootstrap per class)
| Class | observed top-1 | P(stays top-1) | P(top-1 stays in top-3) |
|---|---|---|---|
| Benign | `proto_num` | 1.00 | 1.00 |
| BruteForce | `ssh` | 0.93 | 1.00 |
| DDoS | `num` | 1.00 | 1.00 |
| DoS | `num` | 1.00 | 1.00 |
| Mirai | `tot_size` | 0.89 | 1.00 |
| Recon | `proto_num` | 1.00 | 1.00 |
| Spoofing | `proto_num` | 0.56 | 1.00 |
| Web | `tot_size` | 0.91 | 1.00 |
