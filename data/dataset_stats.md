# Dataset statistics (test split, n=9,000) — provenance for §3.1/§4.3

Computed from data/splits/test.parquet.

| Category | n | ARP=0 frac | Number mean | Number min | Number max |
|---|---:|---:|---:|---:|---:|
| Benign | 696 | 0.823 | 10.00 | 10 | 10 |
| BruteForce | 225 | 0.738 | 10.00 | 10 | 10 |
| DDoS | 3872 | 0.913 | 99.96 | 35 | 100 |
| DoS | 1890 | 0.932 | 99.99 | 89 | 100 |
| Mirai | 1077 | 0.742 | 99.81 | 11 | 100 |
| Recon | 552 | 0.741 | 10.00 | 10 | 10 |
| Spoofing | 464 | 0.858 | 10.00 | 9 | 10 |
| Web | 224 | 0.670 | 10.00 | 10 | 10 |

Overall Spoofing ARP=0 fraction: 0.858. The Number column confirms the two windowing regimes (≈10 vs ≈100 packets/window).
