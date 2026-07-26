# IG vs feature-ablation cross-check (DistilBERT)

Feature ablation: mask each of the 39 fields to [PAD], measure |true-class logit drop|, 60 correctly-classified samples/class. Compared to the Layer-IG rankings (captum_feature_attribution.csv). Agreement => IG is not a method artifact.

| Class | Spearman(all 39) | top-5 overlap | IG top-3 | ablation top-3 |
|---|---|---|---|---|
| Benign | +0.909 | 40% | `proto_num`, `https`, `ttl` | `https`, `ttl`, `rate` |
| BruteForce | +0.850 | 20% | `ssh`, `proto_num`, `psh` | `ssh`, `max`, `irc` |
| DDoS | +0.665 | 40% | `num`, `var`, `syn_cnt` | `proto_num`, `num`, `psh` |
| DoS | +0.843 | 60% | `num`, `var`, `max` | `num`, `var`, `iat` |
| Mirai | +0.834 | 60% | `tot_size`, `max`, `proto_num` | `num`, `proto_num`, `max` |
| Recon | +0.796 | 40% | `proto_num`, `syn`, `ack_cnt` | `max`, `min`, `syn` |
| Spoofing | +0.863 | 100% | `proto_num`, `max`, `min` | `max`, `min`, `https` |
| Web | +0.845 | 60% | `tot_size`, `max`, `avg` | `max`, `rate`, `avg` |

**Mean Spearman +0.826, mean top-5 overlap 52%.** An independent, gradient-free method reproduces the IG feature rankings, confirming the IG attributions are model signal, not a Captum artifact.
