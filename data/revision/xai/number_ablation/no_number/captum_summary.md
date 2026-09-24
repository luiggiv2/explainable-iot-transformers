# DistilBERT `no_number` — Layer Integrated Gradients

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

- **Benign:** `https` (0.1592), `ttl` (0.1493), `ack_cnt` (0.1033), `psh` (0.0975), `proto_num` (0.0810), `dns` (0.0488), `ack` (0.0338), `max` (0.0284)
- **BruteForce:** `ssh` (0.1921), `irc` (0.1085), `ack_cnt` (0.0906), `psh` (0.0822), `max` (0.0712), `ttl` (0.0561), `https` (0.0504), `proto_num` (0.0322)
- **DDoS:** `proto_num` (0.1195), `iat` (0.0667), `ack_cnt` (0.0618), `psh` (0.0535), `rst_cnt` (0.0389), `fin` (0.0384), `ack` (0.0347), `syn_cnt` (0.0344)
- **DoS:** `max` (0.0955), `proto_num` (0.0912), `iat` (0.0806), `tot_size` (0.0723), `avg` (0.0706), `header_len` (0.0514), `rate` (0.0501), `min` (0.0459)
- **Mirai:** `tot_size` (0.2176), `max` (0.2124), `min` (0.1163), `avg` (0.1052), `proto_num` (0.0863), `tot_sum` (0.0509), `iat` (0.0351), `ttl` (0.0223)
- **Recon:** `ttl` (0.1080), `psh` (0.0790), `syn` (0.0758), `syn_cnt` (0.0696), `proto_num` (0.0571), `http` (0.0523), `ack_cnt` (0.0453), `https` (0.0436)
- **Spoofing:** `max` (0.2436), `ttl` (0.1441), `min` (0.1256), `proto_num` (0.0535), `psh` (0.0518), `ack_cnt` (0.0430), `avg` (0.0425), `https` (0.0340)
- **Web:** `ttl` (0.1097), `psh` (0.0993), `rate` (0.0915), `https` (0.0666), `max` (0.0653), `header_len` (0.0564), `ack_cnt` (0.0551), `proto_num` (0.0527)

## Completeness diagnostic

Absolute convergence delta: mean 0.081207, median 0.037147, maximum 1.585442.

The convergence delta is retained per sample so attribution quality can be inspected rather than assumed.
