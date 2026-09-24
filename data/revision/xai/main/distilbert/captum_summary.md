# DistilBERT `original` — Layer Integrated Gradients

The analysis uses 880 fixed cohort rows, 50 integration steps, an all-[PAD] content baseline, and per-sample L1 normalization after mapping wordpieces back to flow fields.

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

- **Benign:** `ttl` (0.1064), `https` (0.0945), `proto_num` (0.0887), `ack_cnt` (0.0748), `ack` (0.0669), `rate` (0.0614), `num` (0.0573), `dns` (0.0535)
- **BruteForce:** `max` (0.0903), `ack_cnt` (0.0707), `avg` (0.0628), `irc` (0.0585), `num` (0.0560), `rate` (0.0543), `ssh` (0.0528), `tot_size` (0.0449)
- **DDoS:** `proto_num` (0.1154), `tot_size` (0.0872), `num` (0.0766), `iat` (0.0616), `syn_cnt` (0.0539), `avg` (0.0509), `ack_cnt` (0.0428), `max` (0.0391)
- **DoS:** `tot_size` (0.1120), `iat` (0.1042), `avg` (0.0860), `max` (0.0643), `rate` (0.0473), `num` (0.0414), `min` (0.0375), `ack_cnt` (0.0362)
- **Mirai:** `proto_num` (0.2702), `tot_size` (0.0683), `max` (0.0478), `avg` (0.0446), `ttl` (0.0369), `tot_sum` (0.0283), `min` (0.0273), `ack_cnt` (0.0268)
- **Recon:** `num` (0.0702), `syn_cnt` (0.0605), `ack_cnt` (0.0599), `https` (0.0543), `rate` (0.0522), `max` (0.0485), `fin` (0.0465), `syn` (0.0460)
- **Spoofing:** `num` (0.1288), `max` (0.1210), `ack_cnt` (0.0912), `min` (0.0658), `https` (0.0633), `ttl` (0.0474), `avg` (0.0459), `dns` (0.0419)
- **Web:** `rate` (0.1243), `ack_cnt` (0.0828), `max` (0.0715), `psh` (0.0715), `num` (0.0709), `avg` (0.0516), `ttl` (0.0506), `tot_size` (0.0487)

## Completeness diagnostic

Absolute convergence delta: mean 0.048675, median 0.024855, maximum 1.185732.

The convergence delta is retained per sample so attribution quality can be inspected rather than assumed.
