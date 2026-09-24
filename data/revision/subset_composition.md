# Subset composition — CICIoT2023 stratified sample

Seed 42; sqrt-proportional category allocation, floor 1500; subtypes proportional within category; nested grouped-stratified split (approximately 5/7, 1/7, 1/7 for train/validation/test). Exact feature duplicates and serialization-equivalent rows are confined to one split. Feature groups use the no-Number/no-Protocol serialization, which also protects all richer registered feature sets.
Malformed rows dropped from raw scan: 9 (of 45,019,243). Rows dropped for non-finite feature values (inf/NaN): 3.

## Per category
| Category   | Subtype   |   Full dataset |   train |   val |   test |
|:-----------|:----------|---------------:|--------:|------:|-------:|
| Benign     | (all)     |        1051373 |    3315 |   663 |    662 |
| BruteForce | (all)     |          12522 |    1071 |   215 |    214 |
| DDoS       | (all)     |       32536197 |   18435 |  3687 |   3687 |
| DoS        | (all)     |        7746554 |    8996 |  1799 |   1800 |
| Mirai      | (all)     |        2521731 |    5132 |  1027 |   1026 |
| Recon      | (all)     |         661121 |    2627 |   525 |    527 |
| Spoofing   | (all)     |         465937 |    2206 |   442 |    441 |
| Web        | (all)     |          23799 |    1071 |   214 |    215 |

## Per subtype
| Category   | Subtype                 |   Full dataset |   train |   val |   test |
|:-----------|:------------------------|---------------:|--------:|------:|-------:|
| Benign     | BENIGN                  |        1051373 |    3315 |   663 |    662 |
| BruteForce | DICTIONARYBRUTEFORCE    |          12522 |    1071 |   215 |    214 |
| DDoS       | DDOS-ACK_FRAGMENTATION  |         272793 |     154 |    31 |     31 |
| DDoS       | DDOS-HTTP_FLOOD         |          27597 |      16 |     3 |      3 |
| DDoS       | DDOS-ICMP_FLOOD         |        6893259 |    3907 |   781 |    781 |
| DDoS       | DDOS-ICMP_FRAGMENTATION |         433157 |     245 |    50 |     49 |
| DDoS       | DDOS-PSHACK_FLOOD       |        3920372 |    2222 |   444 |    444 |
| DDoS       | DDOS-RSTFINFLOOD        |        3872808 |    2194 |   439 |    439 |
| DDoS       | DDOS-SLOWLORIS          |          22400 |      13 |     3 |      2 |
| DDoS       | DDOS-SYNONYMOUSIP_FLOOD |        3445659 |    1951 |   390 |    391 |
| DDoS       | DDOS-SYN_FLOOD          |        3886130 |    2202 |   440 |    441 |
| DDoS       | DDOS-TCP_FLOOD          |        4306086 |    2440 |   488 |    488 |
| DDoS       | DDOS-UDP_FLOOD          |        5181027 |    2935 |   587 |    587 |
| DDoS       | DDOS-UDP_FRAGMENTATION  |         274909 |     156 |    31 |     31 |
| DoS        | DOS-HTTP_FLOOD          |          68799 |      80 |    16 |     16 |
| DoS        | DOS-SYN_FLOOD           |        1942176 |    2255 |   451 |    452 |
| DoS        | DOS-TCP_FLOOD           |        2558256 |    2971 |   594 |    594 |
| DoS        | DOS-UDP_FLOOD           |        3177323 |    3690 |   738 |    738 |
| Mirai      | MIRAI-GREETH_FLOOD      |         949381 |    1931 |   387 |    386 |
| Mirai      | MIRAI-GREIP_FLOOD       |         719655 |    1465 |   293 |    293 |
| Mirai      | MIRAI-UDPPLAIN          |         852695 |    1736 |   347 |    347 |
| Recon      | RECON-HOSTDISCOVERY     |         128677 |     511 |   103 |    102 |
| Recon      | RECON-OSSCAN            |          93970 |     374 |    74 |     75 |
| Recon      | RECON-PINGSWEEP         |           2161 |       8 |     2 |      2 |
| Recon      | RECON-PORTSCAN          |          78730 |     313 |    62 |     63 |
| Recon      | VULNERABILITYSCAN       |         357583 |    1421 |   284 |    285 |
| Spoofing   | DNS_SPOOFING            |         171468 |     812 |   163 |    162 |
| Spoofing   | MITM-ARPSPOOFING        |         294469 |    1394 |   279 |    279 |
| Web        | BACKDOOR_MALWARE        |           3078 |     138 |    28 |     28 |
| Web        | BROWSERHIJACKING        |           5630 |     253 |    50 |     51 |
| Web        | COMMANDINJECTION        |           5168 |     232 |    47 |     47 |
| Web        | SQLINJECTION            |           5022 |     227 |    45 |     45 |
| Web        | UPLOADING_ATTACK        |           1196 |      54 |    11 |     10 |
| Web        | XSS                     |           3705 |     167 |    33 |     34 |
