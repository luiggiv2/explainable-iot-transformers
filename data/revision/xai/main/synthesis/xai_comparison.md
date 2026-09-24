# Revised cross-model XAI comparison

Layer Integrated Gradients and TreeSHAP use the same 880 `SampleID`-aligned rows. Attributions explain the true class and are L1-normalized per sample before aggregation.

| Class | Spearman | Top-5 overlap | IG top feature | SHAP top feature |
|---|---:|---:|---|---|
| Benign | +0.568 | 2/5 | `ttl` | `num` |
| BruteForce | +0.398 | 2/5 | `max` | `ssh` |
| DDoS | +0.468 | 1/5 | `proto_num` | `icmp` |
| DoS | +0.336 | 1/5 | `tot_size` | `num` |
| Mirai | +0.401 | 3/5 | `proto_num` | `tot_sum` |
| Recon | +0.384 | 1/5 | `num` | `num` |
| Spoofing | +0.607 | 3/5 | `num` | `num` |
| Web | +0.552 | 2/5 | `rate` | `num` |

Mean Spearman correlation: +0.464. Mean top-5 overlap: 37.5%.

Agreement between rankings is descriptive rather than a test that one method validates the other. Differences can reflect the models' decision rules as well as the distinct attribution methods. Domain interpretation should be added only after checking the relevant verified fact sheets.
