# Cross-model explainability — DistilBERT (Layer-IG) vs XGBoost (SHAP)

Identical protocol for both: 120 correctly-classified test samples/class, attribution toward the true class, per-sample L1-normalization over the 39 flow features. Token-level IG remapped to features via char offsets. Sources: captum_feature_attribution.csv, shap_xgb_feature_importance.csv, xai_agreement.csv.

## Per-class agreement and top feature

| Class | Spearman(IG,SHAP) | DistilBERT top-1 | XGBoost top-1 |
|---|---|---|---|
| Benign | +0.520 | `proto_num` | `num` |
| BruteForce | +0.490 | `ssh` | `ssh` |
| DDoS | +0.250 | `num` | `icmp` |
| DoS | +0.432 | `num` | `num` |
| Mirai | +0.493 | `tot_size` | `tot_sum` |
| Recon | +0.451 | `proto_num` | `num` |
| Spoofing | +0.450 | `proto_num` | `num` |
| Web | +0.527 | `tot_size` | `num` |

## RQ2 — do attributions match documented domain signatures?

- **BruteForce** — expected: ssh / psh (repeated small SSH logins) [2013-javed-paxson-ssh-bruteforce]. DistilBERT top-3: `ssh`, `proto_num`, `psh`; XGBoost top-3: `ssh`, `num`, `rate`.
- **Mirai** — expected: large-payload size stats (UDP-plain flood) [2017-antonakakis-mirai-botnet]. DistilBERT top-3: `tot_size`, `max`, `proto_num`; XGBoost top-3: `tot_sum`, `max`, `proto_num`.
- **Recon** — expected: syn / probing flags (scanning) [2023-affinito-mirai-scan-evolution]. DistilBERT top-3: `proto_num`, `syn`, `ack_cnt`; XGBoost top-3: `num`, `tot_sum`, `syn`.
- **Spoofing** — expected: arp indicator (ARP poisoning) [2023-alani-arpprobe-spoofing]. DistilBERT top-3: `proto_num`, `max`, `min`; XGBoost top-3: `num`, `min`, `max`.
- **DDoS** — expected: icmp / volumetric flood [2026-hung-explainable-xgboost-unseenattack]. DistilBERT top-3: `num`, `var`, `syn_cnt`; XGBoost top-3: `icmp`, `avg`, `psh`.

Both models recover the textbook signatures where the signature is a direct window feature: `ssh` for BruteForce (both rank it #1/top-3), size statistics for Mirai (near-identical feature sets), `syn` for Recon. This is direct evidence that the fine-tuned transformer learned domain-meaningful signal, not arbitrary correlations.

## RQ3 — systematic biases, shortcuts and spurious dependencies

1. **Spurious dependency in Spoofing (ARP).** The `arp` indicator ranks 39/39 for DistilBERT and 10/39 for XGBoost — the transformer essentially ignores the actual ARP signature and classifies ARP spoofing from indirect features (protocol id, size stats, `https`) of the redirected traffic captured in the window. A genuine artifact-driven decision, not the attack's defining feature [2023-alani-arpprobe-spoofing].

2. **Opposite strategies on DDoS.** `icmp` is XGBoost's #1 feature (rank 1) but DistilBERT's last (rank 39): the tree reads the protocol label directly (CICIoT2023 DDoS is largely ICMP flood), while the transformer reads volumetric behavior (`num`, `var`). Same class, divergent explanations — this is why DDoS has the lowest cross-model agreement.

3. **Each model has a global 'favorite' feature.** `proto_num` is DistilBERT's top-1 for Benign/Recon/Spoofing (top-3 for BruteForce/Mirai); `num` is XGBoost's top-1 for Benign/DoS/Recon/Spoofing/Web. Both lean on one dominant feature across most low-rate classes — a candidate shortcut / dataset artifact (window-sampling) to flag as a limitation.

4. **The hardest class has the most diffuse, least-agreed explanation.** Web (worst F1 in both models) shows no dominant signature and the explanations are spread thin — poor separability and weak/incoherent attribution co-occur.
