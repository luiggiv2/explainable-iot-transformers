# Bootstrap stability of the ig per-class rankings

Each class's cohort rows were resampled with replacement 10,000 times and the mean absolute attribution recomputed. The table reports how often the observed top feature remains first, and how often it remains within the first three.

| Class | n | Observed top-1 | Stays top-1 | Stays top-3 |
|---|---:|---|---:|---:|
| Benign | 120 | `ttl` | 0.83 | 1.00 |
| BruteForce | 93 | `max` | 1.00 | 1.00 |
| DDoS | 120 | `proto_num` | 0.99 | 1.00 |
| DoS | 120 | `tot_size` | 0.91 | 1.00 |
| Mirai | 120 | `proto_num` | 1.00 | 1.00 |
| Recon | 120 | `num` | 0.84 | 1.00 |
| Spoofing | 120 | `num` | 0.67 | 1.00 |
| Web | 67 | `rate` | 1.00 | 1.00 |

A low value means the class's leading feature is decided by which rows entered the cohort and should not be reported as that class's signature. Stability of the ranking says nothing about whether the feature is causally used; it bounds only the sampling error of the estimate.
