# Layer-IG and field-masking comparison

Both analyses use the same fixed cohort. Masking replaces one serialized field at a time with [PAD] tokens and measures the change in the true-class logit. Per-sample changes are L1-normalized before aggregation.

| Class | Spearman | Top-5 overlap | IG top feature | Masking top feature |
|---|---:|---:|---|---|
| Benign | +0.843 | 2/5 | `ttl` | `ttl` |
| BruteForce | +0.781 | 3/5 | `max` | `max` |
| DDoS | +0.835 | 3/5 | `proto_num` | `proto_num` |
| DoS | +0.876 | 3/5 | `tot_size` | `rate` |
| Mirai | +0.709 | 3/5 | `proto_num` | `proto_num` |
| Recon | +0.740 | 2/5 | `num` | `min` |
| Spoofing | +0.868 | 3/5 | `num` | `max` |
| Web | +0.859 | 2/5 | `rate` | `max` |

Mean Spearman: +0.814. Mean top-5 overlap: 2.62/5.

Field masking is a complementary perturbation check, not proof that the gradient attribution is correct. Agreement strengthens a local reading; disagreement should be reported rather than averaged away.
