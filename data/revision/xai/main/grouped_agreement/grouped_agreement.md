# Feature redundancy and group-level IG/SHAP agreement

Train split: 42,853 rows, 39 fields. Spearman correlations and identities are computed on TRAIN only.

## Exact algebraic identities (train)

| Identity | fraction of rows satisfied | max error |
|---|---:|---:|
| Tot size == AVG | 1.0000 | 0.00e+00 |
| Tot sum == Number * AVG | 1.0000 | 2.41e-16 |
| Tot sum / AVG == Number (rows with AVG != 0) | 1.0000 | 2.03e-16 |
| Variance == Std^2 | 1.0000 | 6.07e-15 |
| IPv == LLC | 1.0000 | 0.00e+00 |
| IPv == 1 - ARP | 1.0000 | 2.22e-16 |

`Protocol Type` regressed linearly on the 15 protocol indicators: R^2 = 0.9718. Largest |Spearman| with an indicator: `udp` +0.579, `tcp` -0.428, `icmp` -0.363, `https` -0.099.

## Field pairs with |Spearman| > 0.9 (train)

| a | b | Spearman |
|---|---|---:|
| `rate` | `iat` | -0.997 |
| `fin` | `fin_cnt` | +1.000 |
| `syn` | `syn_cnt` | +0.999 |
| `rst` | `rst_cnt` | +1.000 |
| `ack` | `ack_cnt` | +0.991 |
| `arp` | `ipv` | -1.000 |
| `arp` | `llc` | -1.000 |
| `ipv` | `llc` | +1.000 |
| `max` | `avg` | +0.980 |
| `max` | `tot_size` | +0.980 |
| `avg` | `tot_size` | +1.000 |
| `std` | `var` | +1.000 |

## Groupings

- **G1** (|rho| > 0.9 or exact pairwise identity; 29 units): multi-field groups {`rate`, `iat`}, {`fin`, `fin_cnt`}, {`syn`, `syn_cnt`}, {`rst`, `rst_cnt`}, {`ack`, `ack_cnt`}, {`arp`, `ipv`, `llc`}, {`max`, `avg`, `tot_size`}, {`std`, `var`}; all other fields are singletons.
- **G2** (G1 plus the three-way identity Tot sum = Number x AVG; 27 units): multi-field groups {`rate`, `iat`}, {`fin`, `fin_cnt`}, {`syn`, `syn_cnt`}, {`rst`, `rst_cnt`}, {`ack`, `ack_cnt`}, {`arp`, `ipv`, `llc`}, {`tot_sum`, `max`, `avg`, `tot_size`, `num`}, {`std`, `var`}.

## IG vs SHAP agreement per level

`sum_abs`: group score = sum of members' class-mean |attribution| (primary). `abs_signed_sum`: class mean over samples of |sum of members' signed attribution| (sensitivity). Top-5 overlap chance level = 5/units.

| Level | units | mean Spearman | range | mean top-5 overlap | chance |
|---|---:|---:|---|---:|---:|
| field | 39 | +0.464 | [+0.336, +0.607] | 1.88/5 (37.5%) | 12.8% |
| G1_sum_abs | 29 | +0.661 | [+0.480, +0.777] | 2.50/5 (50.0%) | 17.2% |
| G1_abs_signed_sum | 29 | +0.649 | [+0.472, +0.777] | 2.50/5 (50.0%) | 17.2% |
| G2_sum_abs | 27 | +0.652 | [+0.483, +0.775] | 2.75/5 (55.0%) | 18.5% |
| G2_abs_signed_sum | 27 | +0.642 | [+0.473, +0.755] | 2.62/5 (52.5%) | 18.5% |

## Per class

| Level | Class | Spearman | top-5 overlap | IG top | SHAP top | SHAP exact-zero units |
|---|---|---:|---:|---|---|---:|
| field | Benign | +0.568 | 2/5 | `ttl` | `num` | 10 |
| field | BruteForce | +0.398 | 2/5 | `max` | `ssh` | 8 |
| field | DDoS | +0.468 | 1/5 | `proto_num` | `icmp` | 9 |
| field | DoS | +0.336 | 1/5 | `tot_size` | `num` | 7 |
| field | Mirai | +0.401 | 3/5 | `proto_num` | `tot_sum` | 9 |
| field | Recon | +0.384 | 1/5 | `num` | `num` | 11 |
| field | Spoofing | +0.607 | 3/5 | `num` | `num` | 11 |
| field | Web | +0.552 | 2/5 | `rate` | `num` | 9 |
| G1_sum_abs | Benign | +0.772 | 3/5 | `ack+ack_cnt` | `num` | 3 |
| G1_sum_abs | BruteForce | +0.585 | 2/5 | `max+avg+tot_size` | `ssh` | 3 |
| G1_sum_abs | DDoS | +0.591 | 2/5 | `max+avg+tot_size` | `icmp` | 4 |
| G1_sum_abs | DoS | +0.620 | 3/5 | `max+avg+tot_size` | `num` | 3 |
| G1_sum_abs | Mirai | +0.480 | 2/5 | `proto_num` | `tot_sum` | 5 |
| G1_sum_abs | Recon | +0.722 | 2/5 | `max+avg+tot_size` | `num` | 4 |
| G1_sum_abs | Spoofing | +0.777 | 3/5 | `max+avg+tot_size` | `num` | 5 |
| G1_sum_abs | Web | +0.743 | 3/5 | `max+avg+tot_size` | `num` | 4 |
| G1_abs_signed_sum | Benign | +0.748 | 3/5 | `ack+ack_cnt` | `num` | 3 |
| G1_abs_signed_sum | BruteForce | +0.592 | 2/5 | `max+avg+tot_size` | `ssh` | 3 |
| G1_abs_signed_sum | DDoS | +0.560 | 2/5 | `max+avg+tot_size` | `icmp` | 4 |
| G1_abs_signed_sum | DoS | +0.610 | 2/5 | `max+avg+tot_size` | `num` | 3 |
| G1_abs_signed_sum | Mirai | +0.472 | 2/5 | `proto_num` | `tot_sum` | 5 |
| G1_abs_signed_sum | Recon | +0.692 | 3/5 | `syn+syn_cnt` | `num` | 4 |
| G1_abs_signed_sum | Spoofing | +0.777 | 3/5 | `max+avg+tot_size` | `num` | 5 |
| G1_abs_signed_sum | Web | +0.744 | 3/5 | `max+avg+tot_size` | `num` | 4 |
| G2_sum_abs | Benign | +0.775 | 3/5 | `tot_sum+max+avg+tot_size+num` | `tot_sum+max+avg+tot_size+num` | 3 |
| G2_sum_abs | BruteForce | +0.564 | 2/5 | `tot_sum+max+avg+tot_size+num` | `ssh` | 3 |
| G2_sum_abs | DDoS | +0.563 | 2/5 | `tot_sum+max+avg+tot_size+num` | `tot_sum+max+avg+tot_size+num` | 4 |
| G2_sum_abs | DoS | +0.595 | 2/5 | `tot_sum+max+avg+tot_size+num` | `tot_sum+max+avg+tot_size+num` | 3 |
| G2_sum_abs | Mirai | +0.483 | 2/5 | `proto_num` | `tot_sum+max+avg+tot_size+num` | 5 |
| G2_sum_abs | Recon | +0.749 | 4/5 | `tot_sum+max+avg+tot_size+num` | `tot_sum+max+avg+tot_size+num` | 4 |
| G2_sum_abs | Spoofing | +0.753 | 4/5 | `tot_sum+max+avg+tot_size+num` | `tot_sum+max+avg+tot_size+num` | 5 |
| G2_sum_abs | Web | +0.731 | 3/5 | `tot_sum+max+avg+tot_size+num` | `tot_sum+max+avg+tot_size+num` | 4 |
| G2_abs_signed_sum | Benign | +0.755 | 3/5 | `ack+ack_cnt` | `tot_sum+max+avg+tot_size+num` | 3 |
| G2_abs_signed_sum | BruteForce | +0.571 | 2/5 | `tot_sum+max+avg+tot_size+num` | `ssh` | 3 |
| G2_abs_signed_sum | DDoS | +0.531 | 2/5 | `tot_sum+max+avg+tot_size+num` | `tot_sum+max+avg+tot_size+num` | 4 |
| G2_abs_signed_sum | DoS | +0.593 | 2/5 | `tot_sum+max+avg+tot_size+num` | `tot_sum+max+avg+tot_size+num` | 3 |
| G2_abs_signed_sum | Mirai | +0.473 | 2/5 | `proto_num` | `tot_sum+max+avg+tot_size+num` | 5 |
| G2_abs_signed_sum | Recon | +0.730 | 4/5 | `tot_sum+max+avg+tot_size+num` | `tot_sum+max+avg+tot_size+num` | 4 |
| G2_abs_signed_sum | Spoofing | +0.754 | 3/5 | `tot_sum+max+avg+tot_size+num` | `tot_sum+max+avg+tot_size+num` | 5 |
| G2_abs_signed_sum | Web | +0.728 | 3/5 | `tot_sum+max+avg+tot_size+num` | `tot_sum+max+avg+tot_size+num` | 4 |

## Selected G1 unit ranks (IG rank / SHAP rank)

| Class | Unit | IG rank | IG abs sum | SHAP rank | SHAP abs sum |
|---|---|---:|---:|---:|---:|
| Benign | `proto_num` | 5 | 0.0887 | 23 | 0.0002 |
| Benign | `ssh` | 26 | 0.0030 | 22 | 0.0009 |
| Benign | `arp+ipv+llc` | 16 | 0.0141 | 17 | 0.0048 |
| Benign | `icmp` | 29 | 0.0024 | 19 | 0.0028 |
| Benign | `tot_sum` | 13 | 0.0156 | 11 | 0.0152 |
| Benign | `max+avg+tot_size` | 3 | 0.0997 | 3 | 0.1175 |
| Benign | `num` | 7 | 0.0573 | 1 | 0.4338 |
| BruteForce | `proto_num` | 17 | 0.0198 | 26 | 0.0001 |
| BruteForce | `ssh` | 7 | 0.0528 | 1 | 0.4248 |
| BruteForce | `arp+ipv+llc` | 15 | 0.0226 | 20 | 0.0039 |
| BruteForce | `icmp` | 25 | 0.0044 | 19 | 0.0047 |
| BruteForce | `tot_sum` | 10 | 0.0350 | 12 | 0.0160 |
| BruteForce | `max+avg+tot_size` | 1 | 0.1979 | 4 | 0.0474 |
| BruteForce | `num` | 6 | 0.0560 | 2 | 0.1877 |
| DDoS | `proto_num` | 2 | 0.1154 | 8 | 0.0381 |
| DDoS | `ssh` | 21 | 0.0106 | 24 | 0.0000 |
| DDoS | `arp+ipv+llc` | 20 | 0.0126 | 18 | 0.0033 |
| DDoS | `icmp` | 28 | 0.0055 | 1 | 0.2006 |
| DDoS | `tot_sum` | 12 | 0.0224 | 11 | 0.0196 |
| DDoS | `max+avg+tot_size` | 1 | 0.1772 | 2 | 0.1693 |
| DDoS | `num` | 4 | 0.0766 | 7 | 0.0589 |
| DoS | `proto_num` | 10 | 0.0287 | 22 | 0.0005 |
| DoS | `ssh` | 26 | 0.0064 | 21 | 0.0008 |
| DoS | `arp+ipv+llc` | 15 | 0.0184 | 15 | 0.0127 |
| DoS | `icmp` | 25 | 0.0066 | 20 | 0.0046 |
| DoS | `tot_sum` | 11 | 0.0232 | 11 | 0.0321 |
| DoS | `max+avg+tot_size` | 1 | 0.2623 | 5 | 0.0541 |
| DoS | `num` | 5 | 0.0414 | 1 | 0.2330 |
| Mirai | `proto_num` | 1 | 0.2702 | 3 | 0.0974 |
| Mirai | `ssh` | 21 | 0.0117 | 25 | 0.0000 |
| Mirai | `arp+ipv+llc` | 12 | 0.0258 | 15 | 0.0014 |
| Mirai | `icmp` | 22 | 0.0109 | 19 | 0.0003 |
| Mirai | `tot_sum` | 10 | 0.0283 | 1 | 0.5945 |
| Mirai | `max+avg+tot_size` | 2 | 0.1607 | 2 | 0.1441 |
| Mirai | `num` | 16 | 0.0174 | 8 | 0.0059 |
| Recon | `proto_num` | 14 | 0.0301 | 26 | 0.0000 |
| Recon | `ssh` | 29 | 0.0024 | 15 | 0.0076 |
| Recon | `arp+ipv+llc` | 18 | 0.0126 | 19 | 0.0045 |
| Recon | `icmp` | 28 | 0.0026 | 18 | 0.0051 |
| Recon | `tot_sum` | 15 | 0.0256 | 2 | 0.1445 |
| Recon | `max+avg+tot_size` | 1 | 0.1119 | 6 | 0.0408 |
| Recon | `num` | 6 | 0.0702 | 1 | 0.3320 |
| Spoofing | `proto_num` | 11 | 0.0324 | 23 | 0.0002 |
| Spoofing | `ssh` | 25 | 0.0043 | 16 | 0.0033 |
| Spoofing | `arp+ipv+llc` | 17 | 0.0122 | 9 | 0.0146 |
| Spoofing | `icmp` | 27 | 0.0036 | 20 | 0.0015 |
| Spoofing | `tot_sum` | 13 | 0.0152 | 14 | 0.0055 |
| Spoofing | `max+avg+tot_size` | 1 | 0.2038 | 3 | 0.1530 |
| Spoofing | `num` | 2 | 0.1288 | 1 | 0.3272 |
| Web | `proto_num` | 9 | 0.0389 | 23 | 0.0015 |
| Web | `ssh` | 28 | 0.0033 | 17 | 0.0079 |
| Web | `arp+ipv+llc` | 16 | 0.0162 | 15 | 0.0101 |
| Web | `icmp` | 22 | 0.0057 | 12 | 0.0175 |
| Web | `tot_sum` | 12 | 0.0211 | 6 | 0.0339 |
| Web | `max+avg+tot_size` | 1 | 0.1719 | 3 | 0.0656 |
| Web | `num` | 5 | 0.0709 | 1 | 0.3943 |
