# Spoofing by subtype: ARP presence and attribution ranks

The CICIoT2023 Spoofing category merges MITM-ARPSPOOFING and DNS_SPOOFING. `ARP` is the per-window fraction of packets carrying ARP.

## ARP-field distribution

| Scope | Group | n | ARP = 0 | ARP > 0 | median | p75 | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | Spoofing (all) | 2206 | 86.8% | 13.2% | 0.00 | 0.00 | 0.90 | 0.022 |
| train | MITM-ARPSPOOFING | 1394 | 91.0% | 9.0% | 0.00 | 0.00 | 0.80 | 0.015 |
| train | DNS_SPOOFING | 812 | 79.7% | 20.3% | 0.00 | 0.00 | 0.90 | 0.033 |
| train | All other categories (reference) | 40647 | 86.8% | 13.2% | 0.00 | 0.00 | 0.90 | 0.009 |
| train | Benign (reference) | 3315 | 81.7% | 18.3% | 0.00 | 0.00 | 0.50 | 0.026 |
| test | Spoofing (all) | 441 | 88.2% | 11.8% | 0.00 | 0.00 | 0.40 | 0.018 |
| test | MITM-ARPSPOOFING | 279 | 91.0% | 9.0% | 0.00 | 0.00 | 0.30 | 0.014 |
| test | DNS_SPOOFING | 162 | 83.3% | 16.7% | 0.00 | 0.00 | 0.40 | 0.024 |
| test | All other categories (reference) | 8131 | 86.8% | 13.2% | 0.00 | 0.00 | 0.40 | 0.009 |
| test | Benign (reference) | 662 | 79.0% | 21.0% | 0.00 | 0.00 | 0.40 | 0.028 |
| xai_cohort | Spoofing (all) | 120 | 92.5% | 7.5% | 0.00 | 0.00 | 0.30 | 0.011 |
| xai_cohort | MITM-ARPSPOOFING | 79 | 93.7% | 6.3% | 0.00 | 0.00 | 0.30 | 0.010 |
| xai_cohort | DNS_SPOOFING | 41 | 90.2% | 9.8% | 0.00 | 0.00 | 0.20 | 0.012 |
| xai_cohort | All other categories (reference) | 760 | 81.4% | 18.6% | 0.00 | 0.00 | 0.40 | 0.020 |
| xai_cohort | Benign (reference) | 120 | 85.0% | 15.0% | 0.00 | 0.00 | 0.20 | 0.018 |

## Mean protocol-indicator composition (per-window fractions)

| Scope | Group | ARP | TCP | UDP | DNS | HTTP | HTTPS | ICMP | IPv | LLC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | Spoofing (all) | 0.022 | 0.658 | 0.311 | 0.020 | 0.071 | 0.545 | 0.009 | 0.978 | 0.978 |
| train | MITM-ARPSPOOFING | 0.015 | 0.698 | 0.278 | 0.020 | 0.007 | 0.665 | 0.009 | 0.985 | 0.985 |
| train | DNS_SPOOFING | 0.033 | 0.590 | 0.366 | 0.019 | 0.180 | 0.341 | 0.010 | 0.967 | 0.967 |
| train | All other categories (reference) | 0.009 | 0.555 | 0.246 | 0.009 | 0.051 | 0.121 | 0.106 | 0.991 | 0.991 |
| train | Benign (reference) | 0.026 | 0.809 | 0.156 | 0.030 | 0.052 | 0.695 | 0.009 | 0.974 | 0.974 |
| test | Spoofing (all) | 0.018 | 0.675 | 0.298 | 0.018 | 0.078 | 0.560 | 0.009 | 0.982 | 0.982 |
| test | MITM-ARPSPOOFING | 0.014 | 0.729 | 0.247 | 0.018 | 0.006 | 0.695 | 0.010 | 0.986 | 0.986 |
| test | DNS_SPOOFING | 0.024 | 0.583 | 0.386 | 0.017 | 0.201 | 0.328 | 0.007 | 0.976 | 0.976 |
| test | All other categories (reference) | 0.009 | 0.553 | 0.247 | 0.009 | 0.050 | 0.120 | 0.106 | 0.991 | 0.991 |
| test | Benign (reference) | 0.028 | 0.817 | 0.146 | 0.030 | 0.051 | 0.697 | 0.008 | 0.972 | 0.972 |
| xai_cohort | Spoofing (all) | 0.011 | 0.610 | 0.368 | 0.018 | 0.057 | 0.520 | 0.011 | 0.989 | 0.989 |
| xai_cohort | MITM-ARPSPOOFING | 0.010 | 0.703 | 0.275 | 0.015 | 0.004 | 0.672 | 0.013 | 0.990 | 0.990 |
| xai_cohort | DNS_SPOOFING | 0.012 | 0.432 | 0.549 | 0.024 | 0.159 | 0.227 | 0.007 | 0.988 | 0.988 |
| xai_cohort | All other categories (reference) | 0.020 | 0.587 | 0.216 | 0.017 | 0.059 | 0.220 | 0.054 | 0.980 | 0.980 |
| xai_cohort | Benign (reference) | 0.018 | 0.852 | 0.127 | 0.023 | 0.019 | 0.801 | 0.003 | 0.983 | 0.983 |

## Attribution rank of `arp` and `num` by subtype (XAI cohort, 39 fields)

Ranks are by mean |per-sample L1-normalized attribution| within the group.

| Method | Group | n | rank `arp` | mean abs `arp` | rank `num` | mean abs `num` | top-5 |
|---|---|---:|---:|---:|---:|---:|---|
| Layer-IG (DistilBERT) | Spoofing (all) | 120 | 37 | 0.0031 | 1 | 0.1288 | num, max, ack_cnt, min, https |
| Layer-IG (DistilBERT) | MITM-ARPSPOOFING | 79 | 38 | 0.0031 | 2 | 0.1192 | max, num, ack_cnt, https, ttl |
| Layer-IG (DistilBERT) | DNS_SPOOFING | 41 | 37 | 0.0031 | 1 | 0.1472 | num, max, min, ack_cnt, tot_size |
| TreeSHAP (XGBoost) | Spoofing (all) | 120 | 10 | 0.0146 | 1 | 0.3272 | num, min, max, header_len, iat |
| TreeSHAP (XGBoost) | MITM-ARPSPOOFING | 79 | 10 | 0.0163 | 1 | 0.3428 | num, min, max, header_len, iat |
| TreeSHAP (XGBoost) | DNS_SPOOFING | 41 | 10 | 0.0113 | 2 | 0.2973 | min, num, max, https, iat |
