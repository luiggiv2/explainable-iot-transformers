# Subset composition — CICIoT2023 stratified sample

Seed 42; sqrt-proportional category allocation, floor 1500; subtypes proportional within category; split 70/15/15 stratified by subtype.
Malformed rows dropped from raw scan: 9 (of 45,019,243). Rows dropped for non-finite feature values (inf/NaN): 3.

## Per category
| Category   | Subtype   |   Full dataset |   train |   val |   test |
|:-----------|:----------|---------------:|--------:|------:|-------:|
| Benign     | (all)     |        1051373 |    3248 |   696 |    696 |
| BruteForce | (all)     |          12522 |    1050 |   225 |    225 |
| DDoS       | (all)     |       32536197 |   18065 |  3872 |   3872 |
| DoS        | (all)     |        7746554 |    8816 |  1889 |   1890 |
| Mirai      | (all)     |        2521731 |    5030 |  1078 |   1077 |
| Recon      | (all)     |         661121 |    2575 |   552 |    552 |
| Spoofing   | (all)     |         465937 |    2162 |   463 |    464 |
| Web        | (all)     |          23799 |    1051 |   225 |    224 |

## Per subtype
| Category   | Subtype                 |   Full dataset |   train |   val |   test |
|:-----------|:------------------------|---------------:|--------:|------:|-------:|
| Benign     | BENIGN                  |        1051373 |    3248 |   696 |    696 |
| BruteForce | DICTIONARYBRUTEFORCE    |          12522 |    1050 |   225 |    225 |
| DDoS       | DDOS-ACK_FRAGMENTATION  |         272793 |     151 |    33 |     32 |
| DDoS       | DDOS-HTTP_FLOOD         |          27597 |      15 |     4 |      3 |
| DDoS       | DDOS-ICMP_FLOOD         |        6893259 |    3828 |   820 |    821 |
| DDoS       | DDOS-ICMP_FRAGMENTATION |         433157 |     241 |    51 |     52 |
| DDoS       | DDOS-PSHACK_FLOOD       |        3920372 |    2177 |   467 |    466 |
| DDoS       | DDOS-RSTFINFLOOD        |        3872808 |    2150 |   461 |    461 |
| DDoS       | DDOS-SLOWLORIS          |          22400 |      13 |     2 |      3 |
| DDoS       | DDOS-SYNONYMOUSIP_FLOOD |        3445659 |    1912 |   410 |    410 |
| DDoS       | DDOS-SYN_FLOOD          |        3886130 |    2158 |   463 |    462 |
| DDoS       | DDOS-TCP_FLOOD          |        4306086 |    2391 |   512 |    513 |
| DDoS       | DDOS-UDP_FLOOD          |        5181027 |    2876 |   616 |    617 |
| DDoS       | DDOS-UDP_FRAGMENTATION  |         274909 |     153 |    33 |     32 |
| DoS        | DOS-HTTP_FLOOD          |          68799 |      78 |    17 |     17 |
| DoS        | DOS-SYN_FLOOD           |        1942176 |    2211 |   473 |    474 |
| DoS        | DOS-TCP_FLOOD           |        2558256 |    2911 |   624 |    624 |
| DoS        | DOS-UDP_FLOOD           |        3177323 |    3616 |   775 |    775 |
| Mirai      | MIRAI-GREETH_FLOOD      |         949381 |    1893 |   406 |    405 |
| Mirai      | MIRAI-GREIP_FLOOD       |         719655 |    1436 |   307 |    308 |
| Mirai      | MIRAI-UDPPLAIN          |         852695 |    1701 |   365 |    364 |
| Recon      | RECON-HOSTDISCOVERY     |         128677 |     501 |   107 |    108 |
| Recon      | RECON-OSSCAN            |          93970 |     366 |    79 |     78 |
| Recon      | RECON-PINGSWEEP         |           2161 |       8 |     2 |      2 |
| Recon      | RECON-PORTSCAN          |          78730 |     307 |    66 |     65 |
| Recon      | VULNERABILITYSCAN       |         357583 |    1393 |   298 |    299 |
| Spoofing   | DNS_SPOOFING            |         171468 |     796 |   170 |    171 |
| Spoofing   | MITM-ARPSPOOFING        |         294469 |    1366 |   293 |    293 |
| Web        | BACKDOOR_MALWARE        |           3078 |     136 |    29 |     29 |
| Web        | BROWSERHIJACKING        |           5630 |     248 |    53 |     53 |
| Web        | COMMANDINJECTION        |           5168 |     228 |    49 |     49 |
| Web        | SQLINJECTION            |           5022 |     222 |    48 |     47 |
| Web        | UPLOADING_ATTACK        |           1196 |      53 |    11 |     11 |
| Web        | XSS                     |           3705 |     164 |    35 |     35 |
