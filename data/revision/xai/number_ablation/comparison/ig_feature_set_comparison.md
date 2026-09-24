# Layer-IG comparison after removing `Number`

Both models are explained on the same rows, and every row is correctly classified by both checkpoints. Correlations use the 38 fields shared by the two serializations.

| Class | Shared-feature Spearman | Original `num` rank | Original `num` magnitude | Largest increase without `num` |
|---|---:|---:|---:|---|
| Benign | +0.820 | 7 | 0.0515 | `https` (+0.0511) |
| BruteForce | +0.833 | 6 | 0.0543 | `ssh` (+0.1294) |
| DDoS | +0.582 | 3 | 0.0625 | `rst_cnt` (+0.0245) |
| DoS | +0.816 | 5 | 0.0467 | `proto_num` (+0.0631) |
| Mirai | +0.541 | 11 | 0.0223 | `max` (+0.1492) |
| Recon | +0.847 | 2 | 0.0719 | `ttl` (+0.0628) |
| Spoofing | +0.909 | 2 | 0.1377 | `max` (+0.1051) |
| Web | +0.905 | 2 | 0.0797 | `ttl` (+0.0561) |

Because each sample is L1-normalized, increases after removing `Number` describe redistributed attribution mass; they are not independent effect sizes. The performance ablation remains the stronger evidence about the predictive consequence of removing the field.
