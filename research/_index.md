# Research fact-sheet index

All four clusters covered: **1** (transformers/SLMs for traffic classification), **2** (ML-IDS baselines on CICIoT2023 & comparable datasets), **3** (XAI for intrusion detection), **4** (attack-signature domain knowledge).

Note: the Biswas et al. 2026 paper was surfaced independently by the cluster 1 and cluster 3 searches; the two fact sheets were merged into `2026-biswas-attribution-encoderllm-ids.md` (cluster 1, 3).

| File | Cluster | Year | Dataset(s) | Verified DOI |
|---|---|---|---|---|
| [2022-lin-etbert.md](2022-lin-etbert.md) | 1 | 2022 | ISCX-Tor, ISCX-VPN, Cross-Platform, CSTNET-TLS 1.3 | yes |
| [2023-hegselmann-tabllm.md](2023-hegselmann-tabllm.md) | 1 | 2023 | 9 public tabular benchmarks + 3 healthcare tasks (confirmed) | none (PMLR v206 — no article DOI; official citation confirmed) |
| [2023-meng-netgpt.md](2023-meng-netgpt.md) | 1 | 2023 | Self-collected traffic (encrypted, DNS, industrial, mining) | no (arXiv only) |
| [2023-shi-tsfn-bertlstm.md](2023-shi-tsfn-bertlstm.md) | 1 | 2023 | USTC-TFC | yes |
| [2025-afifi-mindiot-tokenization.md](2025-afifi-mindiot-tokenization.md) | 1 | 2025 | UNSW IoT Traces, MonIoTr, IoT Sentinel, YourThings, IoT-FCSIT | yes |
| [2025-almadhor-otad-iot.md](2025-almadhor-otad-iot.md) | 1 | 2025 | RT_IoT2022 | yes |
| [2025-cui-trafficllm.md](2025-cui-trafficllm.md) | 1 | 2025 | 10 open traffic datasets `[VERIFICAR]` | no (arXiv only) |
| [2025-tang-mtfbert-federated.md](2025-tang-mtfbert-federated.md) | 1 | 2025 | USTC-TFC2016, IoT-22, IoT-23, CIC-MalAnal2017 | yes |
| [2026-algarni-lightweightbert-iot.md](2026-algarni-lightweightbert-iot.md) | 1 | 2026 | ToN_IoT, Edge-IIoTset, X-IIoTID | yes |
| [2026-biswas-attribution-encoderllm-ids.md](2026-biswas-attribution-encoderllm-ids.md) | 1, 3 | 2026 | CICIDS2017 (SDN framing) | yes (arXiv DOI, preprint) |
| [2026-martinreizabal-flowtowindow-ciciot.md](2026-martinreizabal-flowtowindow-ciciot.md) | 1 | 2026 | **CICIoT2023 only** (6-class subset: Benign+DDoS-ICMP+DoS-TCP+DDoS-TCP+OSScan+PortScan; confirmed from primary PDF — Bot-IoT/TON_IoT/KDD-99 NOT used, they only appear in the paper's related-work comparison table). Architecture confirmed: small custom Transformer encoder (d=64, N=4, H=4), NOT a pretrained LM. Metrics (97.9% weighted-F1 / 99.6% macro ROC-AUC) confirmed from Table 6 of primary PDF — metric-type caveat recorded (not macro-F1, not comparable to our 0.707 without explicit unit conversion). | yes (fully verified from primary PDF) |
| [2026-mayhoub-talklikeapacket-foundation.md](2026-mayhoub-talklikeapacket-foundation.md) | 1 | 2026 | N/A (perspective paper) | yes — published: IEEE Comm. Magazine 64(5), 10.1109/MCOM.001.2500414 |
| [2023-neto-ciciot2023-dataset.md](2023-neto-ciciot2023-dataset.md) | 2 | 2023 | **CICIoT2023** (original dataset paper — mandatory citation) | yes |
| [2025-adewole-ruleinduction-ciciot2023.md](2025-adewole-ruleinduction-ciciot2023.md) | 2 | 2025 | CIC-IDS2017, **CICIoT2023** | yes |
| [2025-alharby-mlefficiency-ciciot2023.md](2025-alharby-mlefficiency-ciciot2023.md) | 2 | 2025 | **CICIoT2023** (balanced 4-class subset, 302,896/class) | yes |
| [2025-bahmani-mloptimization-ciciot2023.md](2025-bahmani-mloptimization-ciciot2023.md) | 2 | 2025 | **CICIoT2023** (1.19M-sample subset, 6 classes; fully verified from publisher XML) | yes |
| [2025-khan-woxgb-conceptdrift.md](2025-khan-woxgb-conceptdrift.md) | 2 | 2025 | Edge-IIoTset, **CICIoT2023** | yes |
| [2025-shaikhanova-securityaudit-tonIoT.md](2025-shaikhanova-securityaudit-tonIoT.md) | 2 | 2025 | TON_IoT | yes |
| [2026-hung-explainable-xgboost-unseenattack.md](2026-hung-explainable-xgboost-unseenattack.md) | 2 | 2026 | **CICIoT2023** (2M-sample training subset; unseen-attack protocol) | yes |
| [2026-moucharraf-autoboost-hybrid.md](2026-moucharraf-autoboost-hybrid.md) | 2 | 2026 | N-BaIoT, **CICIoT2023** (11-attack subset) | yes |
| [2025-alnomasy-distillguard-transformer.md](2025-alnomasy-distillguard-transformer.md) | 3 | 2025 | **ToN-IoT, RT-IoT (RT-IoT2022), N-BaIoT** (all three confirmed from primary PDF Section 4.5; per-dataset/per-task metrics and parameter counts fully confirmed, Tables 5-11) | yes |
| [2025-bacevicius-lime-perturbation.md](2025-bacevicius-lime-perturbation.md) | 3 | 2025 | **CIC-IDS2017 + CSE-CIC-IDS-2018, merged** (28 classes, 32 features; confirmed from primary PDF Section 3). Model list confirmed: DT, RF, K-NN, XGBoost, CatBoost. | yes |
| [2025-hermosilla-forensic-shap-lime.md](2025-hermosilla-forensic-shap-lime.md) | 3 | 2025 | UNSW-NB15 | yes |
| [2025-khan-industry5-adversarial-xai-review.md](2025-khan-industry5-adversarial-xai-review.md) | 3 | 2025 | N/A (systematic review, 135 studies) | yes |
| [2025-nugraha-versatile-xai-framework.md](2025-nugraha-versatile-xai-framework.md) | 3 | 2025 | CIC-DDoS2019, **CICIoT2023**, 5G PFCP | yes |
| [2025-obidiagha-iforestexplain-ciciot2023.md](2025-obidiagha-iforestexplain-ciciot2023.md) | 3 | 2025 | **CICIoT2023** | yes |
| [2026-bouke-entropy-explainable-nid.md](2026-bouke-entropy-explainable-nid.md) | 3 | 2026 | NSL-KDD, CICIDS2017/2018, UNSW-NB15 | yes (arXiv DOI, preprint) |
| [2026-dong-ngboost-shap-iot.md](2026-dong-ngboost-shap-iot.md) | 3 | 2026 | UNSW-NB15, CICIDS2017, N-BaIoT (confirmed via convergent sources) | yes |
| [2026-rajhans-explainability-stability.md](2026-rajhans-explainability-stability.md) | 3 | 2026 | Phishing URLs, UNSW-NB15, NF-ToN-IoT, HIKARI-2021 | yes (arXiv DOI, preprint) |
| [2026-sheikhi-exai5g-transformer.md](2026-sheikhi-exai5g-transformer.md) | 3 | 2026 | **Custom, self-collected 5G/IoT testbed dataset** (confirmed NOT a named public benchmark; synthetically generated attacks, 9 classes, 1.75M train / 194,829 test records — confirmed from primary PDF Section 4 + Ethics statement) | yes (arXiv DOI, preprint — still preprint-only, re-checked) |
| [2026-vourganas-multicollinearity-fragility.md](2026-vourganas-multicollinearity-fragility.md) | 3 | 2026 | UNSW-NB15 | yes (arXiv DOI, preprint) |
| [2013-javed-paxson-ssh-bruteforce.md](2013-javed-paxson-ssh-bruteforce.md) | 4 | 2013 | 8 years of LBNL SSH authentication logs | yes |
| [2017-antonakakis-mirai-botnet.md](2017-antonakakis-mirai-botnet.md) | 4 | 2017 | Multi-vantage Mirai measurement corpus (telescope, honeypots, DNS, C2 logs) | none (confirmed via usenix.org — cite by URL; ISBN 978-1-931971-40-9) |
| [2018-zakroum-networktelescope-portprobing.md](2018-zakroum-networktelescope-portprobing.md) | 4 | 2018 | /20 network telescope, 2014–2017 (~4.5B TCP SYN packets) | yes |
| [2023-affinito-mirai-scan-evolution.md](2023-affinito-mirai-scan-evolution.md) | 4 | 2023 | Darknet capture of Mirai-signature scans, 2016–2022 | yes |
| [2023-alani-arpprobe-spoofing.md](2023-alani-arpprobe-spoofing.md) | 4 | 2023 | **Two named datasets, confirmed from primary PDF**: (1) "IoT-ID" / IoT Network Intrusion Dataset (Kang et al. 2019); (2) SDN/MUD-monitoring dataset (Hamza et al., ACM SOSR 2019) | yes |
| [2025-alkhazaali-nmap-portscan.md](2025-alkhazaali-nmap-portscan.md) | 4 | 2025 | Self-generated Nmap (SYN/Connect only) vs. Metasploit captures, 2-VM testbed (verified from full PDF) | yes |
| [2025-salman-ciciot2023-limitations.md](2025-salman-ciciot2023-limitations.md) | 4 | 2025 | Narrative review of **CICIoT2023** (no new experiments) | yes |

**Totals**: 38 fact sheets — cluster 1: 12 (one shared with cluster 3), cluster 2: 8, cluster 3: 12 (one shared with cluster 1), cluster 4: 7.

## Validation pass — status (2026-07-14)
Four parallel verification batches completed (Elsevier-blocked papers, arXiv preprint status, cluster-4 details, journal quartiles).

**Resolved:**
- Bahmani 2025: fully verified from publisher JATS XML — per-class P/R/F1 for all 8 models recovered; class count corrected (6 classes, not 5: sheet had omitted DoS as distinct from DDoS).
- Alnomasy/DistillGuard: **DOI corrected** — was `10.1016/j.cose.2024.104417` (404s), correct is `10.1016/j.cose.2025.104417`.
- Mayhoub: published — IEEE Communications Magazine 64(5), pp. 36–42, DOI 10.1109/MCOM.001.2500414.
- Hegselmann/TabLLM: official PMLR citation confirmed (v206, pp. 5549–5581; no article DOI exists).
- Antonakakis/Mirai: no DOI exists (USENIX) — cite via usenix.org URL; ISBN and OA PDF link confirmed.
- Al-Khazaali: full PDF read — only SYN/Connect scans actually tested; Scopus/DOAJ-indexed but modest venue → treat as supporting citation only. Stronger alternatives (verified DOIs, no sheets yet): Liao et al. 2020 (10.1109/CyberC49757.2020.00020), Sharafaldin et al. 2018 (10.5220/0006639801080116).
- Preprint status stamped on all arXiv sheets: Rajhans, Vourganas, Bouke, Biswas, Sheikhi, NetGPT, TrafficLLM remain preprint-only as of 2026-07-14 (checked arXiv journal-ref, DBLP, Semantic Scholar). Note: Rajhans's *companion* paper is ICAART 2026-accepted, but that acceptance does not cover the Rajhans sheet's paper.

**Still open (needs institutional access / manual action):**
- **Journal quartiles**: scimagojr.com and doaj.org are network-blocked in this environment (HTTP 403 / org policy) — verify SJR quartiles manually from an unrestricted browser for the ~20 venues. Unverified aggregator hints (NOT confirmed): Sensors Q1, Future Internet Q2, Scientific Reports Q1, Discover IoT Q2 (2024 data). Also still open: `2023-alani-arpprobe-spoofing.md` exact IF/quartile figure and `2025-bacevicius-lime-perturbation.md` quartile (both `[VERIFICAR]`, Scimago-dependent).
- `2026-dong-ngboost-shap-iot.md`: SHAP plot-type enumeration `[VERIFICAR]` (out of scope for this pass — not one of the 5 PDFs provided).

## Validation pass — primary-PDF resolution (2026-07-23)
The author supplied 5 previously paywalled PDFs (now in `research/`), enabling full-text (not just secondary/abstract) resolution of all remaining `[VERIFICAR]` items flagged in the 2026-07-14 pass for these 5 papers:

- `2026-martinreizabal-flowtowindow-ciciot.md`: **Resolved.** Dataset confirmed as CICIoT2023 only (a 6-class subset: Benign + DDoS-ICMP + DoS-TCP + DDoS-TCP + OSScan + PortScan) — Bot-IoT/TON_IoT/KDD-99 do NOT appear in this work's own experiments, only in its related-work comparison table (Table 1). Encoder architecture confirmed: a small custom Transformer encoder trained from scratch on tabular per-flow features (d=64, N=4 layers, H=4 heads, d_ff=128, context length L=127), not a pretrained language model. Metrics (97.9% weighted F1, 99.6% macro ROC-AUC, 95.35% balanced accuracy) confirmed from Table 6 of the primary PDF. Metric-type caveat explicitly documented in the fact sheet (weighted F1 / macro ROC-AUC, NOT macro-F1 — not directly comparable to our own 0.707 macro-F1).
- `2025-alnomasy-distillguard-transformer.md`: **Resolved.** Three datasets confirmed (ToN-IoT, RT-IoT/RT-IoT2022, N-BaIoT), each with full per-class/per-task accuracy, F1, and parameter-count tables (Tables 5-11 of the primary PDF) — including student-model compression ratios up to >200x fewer parameters than the teacher, directly useful for RQ5.
- `2023-alani-arpprobe-spoofing.md`: **Resolved.** The two evaluation datasets are named: (1) the "IoT-ID" / IoT Network Intrusion Dataset (Kang et al. 2019), and (2) the SDN/MUD-monitoring dataset from Hamza, Gharakheili, Benson & Sivaraman (ACM SOSR 2019). Stated limitations (packet-level bottleneck at high traffic volume; resource-constrained on-device deployment) also recovered from the primary PDF's Discussion section.
- `2026-sheikhi-exai5g-transformer.md`: **Resolved.** The "5G/IoT intrusion dataset" is confirmed to be a custom, self-collected, synthetically generated dataset from the authors' own controlled 5G testbed — NOT a named public benchmark (no external dataset citation is given anywhere in the paper; the Ethics statement and the paper's own stated "Dataset Scope" limitation both confirm this framing).
- `2025-bacevicius-lime-perturbation.md`: **Resolved/confirmed.** Dataset is a merged CIC-IDS2017 + CSE-CIC-IDS-2018 dataset (28 classes, 32 features) — refining the task brief's shorthand "CIC-IDS-2018" (the paper's own Conclusion uses that shorthand, but Section 3 clarifies the actual working dataset is the merge). Full model list confirmed: Decision Tree, Random Forest, K-NN, XGBoost, CatBoost (not just KNN/DT). Fidelity (R²) and stability (Jaccard S) tables fully recovered.

All 5 updated fact sheets now cite the primary PDF filename + DOI directly. See `references/verification-log.md` for the corresponding refs.bib verification entries (no DOI/metadata changes resulted from this pass — all 5 DOIs were already correctly recorded and Crossref-verified; this pass only resolved dataset/architecture/metric-detail `[VERIFICAR]` markers using full-text evidence).

**Preprint re-check (2026-07-23):** Re-verified via arXiv abstract-page `journal-ref`/comments fields (direct fetch, not just search) plus WebSearch for each of the 7 arXiv-only papers flagged in the 2026-07-14 pass. **All 7 remain preprint-only — no change in status:**
- `2026-rajhans-explainability-stability.md` (arXiv:2607.01679) — v1 only (2 Jul 2026), no journal-ref.
- `2026-vourganas-multicollinearity-fragility.md` (arXiv:2605.22529) — v1 only (21 May 2026), comments field confirms "submitted to ACM TAISAP" but no acceptance/journal-ref yet.
- `2026-bouke-entropy-explainable-nid.md` (arXiv:2606.29797) — v1 only (29 Jun 2026), no journal-ref.
- `2026-biswas-attribution-encoderllm-ids.md` (arXiv:2604.06266) — v1 only (7 Apr 2026), no journal-ref.
- `2026-sheikhi-exai5g-transformer.md` (arXiv:2604.18052) — v1 only (20 Apr 2026), no journal-ref. (A June 2026 *Telecom* (MDPI) paper cites ExAI5G but this is a citing work, not ExAI5G's own publication.)
- `2023-meng-netgpt.md` (arXiv:2304.09513) — now at v3 (28 Aug 2025, a content update, not a publication), still no journal-ref after 3+ years on arXiv.
- `2025-cui-trafficllm.md` (arXiv:2504.04222) — v2 (15 Apr 2025), no journal-ref.
No refs.bib changes required for these 7 entries.
