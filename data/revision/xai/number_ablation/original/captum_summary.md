# DistilBERT `original` — Layer Integrated Gradients

The analysis uses 480 fixed cohort rows, 50 integration steps, an all-[PAD] content baseline, and per-sample L1 normalization after mapping wordpieces back to flow fields.

## Cohort size

| class      |   rows |
|:-----------|-------:|
| Benign     |     60 |
| BruteForce |     60 |
| DDoS       |     60 |
| DoS        |     60 |
| Mirai      |     60 |
| Recon      |     60 |
| Spoofing   |     60 |
| Web        |     60 |

## Top features by class

- **Benign:** `https` (0.1081), `ttl` (0.1079), `proto_num` (0.0831), `ack_cnt` (0.0759), `rate` (0.0692), `ack` (0.0554), `num` (0.0515), `dns` (0.0502)
- **BruteForce:** `max` (0.0847), `irc` (0.0749), `ack_cnt` (0.0724), `ssh` (0.0627), `avg` (0.0582), `num` (0.0543), `rate` (0.0504), `tot_size` (0.0461)
- **DDoS:** `proto_num` (0.1341), `tot_size` (0.0956), `num` (0.0625), `ack_cnt` (0.0544), `psh` (0.0506), `avg` (0.0496), `iat` (0.0478), `max` (0.0428)
- **DoS:** `iat` (0.1013), `tot_size` (0.1010), `avg` (0.0803), `max` (0.0668), `num` (0.0467), `rate` (0.0439), `min` (0.0367), `ack_cnt` (0.0355)
- **Mirai:** `proto_num` (0.2177), `tot_size` (0.0916), `max` (0.0633), `avg` (0.0534), `ttl` (0.0373), `min` (0.0363), `tot_sum` (0.0362), `ack_cnt` (0.0257)
- **Recon:** `syn_cnt` (0.0751), `num` (0.0719), `rate` (0.0610), `https` (0.0600), `ack_cnt` (0.0578), `ttl` (0.0452), `syn` (0.0450), `max` (0.0441)
- **Spoofing:** `max` (0.1385), `num` (0.1377), `ack_cnt` (0.1032), `https` (0.0640), `avg` (0.0482), `ttl` (0.0466), `min` (0.0464), `dns` (0.0352)
- **Web:** `rate` (0.1314), `num` (0.0797), `ack_cnt` (0.0727), `psh` (0.0681), `max` (0.0573), `ttl` (0.0536), `header_len` (0.0504), `avg` (0.0489)

## Completeness diagnostic

Absolute convergence delta: mean 0.040815, median 0.022530, maximum 0.410674.

The convergence delta is retained per sample so attribution quality can be inspected rather than assumed.
