# XGBoost `original` — TreeSHAP

TreeSHAP was evaluated on 880 fixed cohort rows. Values explain the true class and are L1-normalized within each sample before class-level aggregation.

## Cohort size

| class      |   rows |
|:-----------|-------:|
| Benign     |    120 |
| BruteForce |     93 |
| DDoS       |    120 |
| DoS        |    120 |
| Mirai      |    120 |
| Recon      |    120 |
| Spoofing   |    120 |
| Web        |     67 |

## Top features by class

- **Benign:** `num` (0.4338), `https` (0.1691), `max` (0.1019), `ttl` (0.0431), `psh` (0.0352), `iat` (0.0305), `header_len` (0.0223), `rate` (0.0211)
- **BruteForce:** `ssh` (0.4248), `num` (0.1877), `iat` (0.0977), `max` (0.0410), `https` (0.0308), `http` (0.0216), `header_len` (0.0198), `std` (0.0192)
- **DDoS:** `icmp` (0.2006), `avg` (0.1625), `fin` (0.1156), `psh` (0.1029), `iat` (0.0642), `num` (0.0589), `rate` (0.0407), `syn_cnt` (0.0394)
- **DoS:** `num` (0.2330), `header_len` (0.1769), `iat` (0.0560), `syn` (0.0548), `fin` (0.0518), `min` (0.0491), `psh` (0.0465), `avg` (0.0438)
- **Mirai:** `tot_sum` (0.5945), `proto_num` (0.0974), `max` (0.0819), `min` (0.0753), `avg` (0.0623), `std` (0.0433), `header_len` (0.0091), `tcp` (0.0090)
- **Recon:** `num` (0.3320), `tot_sum` (0.1445), `syn` (0.1357), `rst` (0.0747), `fin` (0.0440), `https` (0.0359), `http` (0.0340), `std` (0.0329)
- **Spoofing:** `num` (0.3272), `min` (0.2809), `max` (0.1424), `header_len` (0.0353), `iat` (0.0349), `https` (0.0333), `ttl` (0.0234), `rate` (0.0167)
- **Web:** `num` (0.3943), `iat` (0.1586), `https` (0.0487), `max` (0.0479), `ttl` (0.0437), `tot_sum` (0.0339), `min` (0.0292), `http` (0.0287)
