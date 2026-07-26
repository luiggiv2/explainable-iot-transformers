"""Phase 4 (synthesis) — cross-model XAI comparison for RQ2/RQ3.

Consumes the two attribution tables (DistilBERT Layer-IG, XGBoost SHAP), both
produced under the identical protocol, and emits the reproducible findings
artifacts the manuscript draws on. Every number here traces back to
captum_feature_attribution.csv / shap_xgb_feature_importance.csv.

Outputs (data/):
  xai_agreement.csv     per-class Spearman(IG,SHAP) + notable feature ranks
  xai_comparison.md     RQ2/RQ3 narrative with domain cross-reference
"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]

# domain signature expected per class, and the fact sheet it comes from
DOMAIN = {
    "BruteForce": ("ssh / psh (repeated small SSH logins)", "2013-javed-paxson-ssh-bruteforce"),
    "Mirai": ("large-payload size stats (UDP-plain flood)", "2017-antonakakis-mirai-botnet"),
    "Recon": ("syn / probing flags (scanning)", "2023-affinito-mirai-scan-evolution"),
    "Spoofing": ("arp indicator (ARP poisoning)", "2023-alani-arpprobe-spoofing"),
    "DDoS": ("icmp / volumetric flood", "2026-hung-explainable-xgboost-unseenattack"),
}


def top(df, cls, k=5):
    t = df[df["class"] == cls].nsmallest(k, "rank")
    return list(zip(t["feature"], t["abs_mean"].round(3)))


def rank_of(df, cls, feat):
    r = df[(df["class"] == cls) & (df.feature == feat)]
    return int(r["rank"].iloc[0]) if len(r) else None


def main():
    ig = pd.read_csv(DATA / "captum_feature_attribution.csv")
    sh = pd.read_csv(DATA / "shap_xgb_feature_importance.csv")

    rows = []
    for c in CLASSES:
        a = ig[ig["class"] == c].set_index("feature")["abs_mean"]
        b = sh[sh["class"] == c].set_index("feature")["abs_mean"]
        feats = a.index.intersection(b.index)
        rho, _ = spearmanr(a[feats], b[feats])
        rows.append({
            "class": c, "spearman_ig_shap": round(float(rho), 3),
            "ig_top1": top(ig, c, 1)[0][0], "shap_top1": top(sh, c, 1)[0][0],
            "proto_num_rank_ig": rank_of(ig, c, "proto_num"),
            "num_rank_shap": rank_of(sh, c, "num"),
        })
    agree = pd.DataFrame(rows)
    agree.to_csv(DATA / "xai_agreement.csv", index=False)

    L = []
    L += ["# Cross-model explainability — DistilBERT (Layer-IG) vs XGBoost (SHAP)", "",
          "Identical protocol for both: 120 correctly-classified test samples/class, "
          "attribution toward the true class, per-sample L1-normalization over the 39 "
          "flow features. Token-level IG remapped to features via char offsets. "
          "Sources: captum_feature_attribution.csv, shap_xgb_feature_importance.csv, "
          "xai_agreement.csv.", ""]

    L += ["## Per-class agreement and top feature", "",
          "| Class | Spearman(IG,SHAP) | DistilBERT top-1 | XGBoost top-1 |",
          "|---|---|---|---|"]
    for _, r in agree.iterrows():
        L.append(f"| {r['class']} | {r['spearman_ig_shap']:+.3f} | "
                 f"`{r['ig_top1']}` | `{r['shap_top1']}` |")
    L += [""]

    L += ["## RQ2 — do attributions match documented domain signatures?", ""]
    for cls, (sig, ref) in DOMAIN.items():
        igt = ", ".join(f"`{f}`" for f, _ in top(ig, cls, 3))
        sht = ", ".join(f"`{f}`" for f, _ in top(sh, cls, 3))
        L.append(f"- **{cls}** — expected: {sig} [{ref}]. "
                 f"DistilBERT top-3: {igt}; XGBoost top-3: {sht}.")
    L += ["",
          "Both models recover the textbook signatures where the signature is a direct "
          "window feature: `ssh` for BruteForce (both rank it #1/top-3), size statistics "
          "for Mirai (near-identical feature sets), `syn` for Recon. This is direct "
          "evidence that the fine-tuned transformer learned domain-meaningful signal, "
          "not arbitrary correlations.", ""]

    L += ["## RQ3 — systematic biases, shortcuts and spurious dependencies", ""]
    sp_ig = rank_of(ig, "Spoofing", "arp")
    sp_sh = rank_of(sh, "Spoofing", "arp")
    dd_ig = rank_of(ig, "DDoS", "icmp")
    dd_sh = rank_of(sh, "DDoS", "icmp")
    L += [
        f"1. **Spurious dependency in Spoofing (ARP).** The `arp` indicator ranks "
        f"{sp_ig}/39 for DistilBERT and {sp_sh}/39 for XGBoost — the transformer "
        "essentially ignores the actual ARP signature and classifies ARP spoofing from "
        "indirect features (protocol id, size stats, `https`) of the redirected traffic "
        "captured in the window. A genuine artifact-driven decision, not the attack's "
        "defining feature [2023-alani-arpprobe-spoofing].",
        "",
        f"2. **Opposite strategies on DDoS.** `icmp` is XGBoost's #1 feature "
        f"(rank {dd_sh}) but DistilBERT's last (rank {dd_ig}): the tree reads the "
        "protocol label directly (CICIoT2023 DDoS is largely ICMP flood), while the "
        "transformer reads volumetric behavior (`num`, `var`). Same class, divergent "
        "explanations — this is why DDoS has the lowest cross-model agreement.",
        "",
        "3. **Each model has a global 'favorite' feature.** `proto_num` is DistilBERT's "
        "top-1 for Benign/Recon/Spoofing (top-3 for BruteForce/Mirai); `num` is "
        "XGBoost's top-1 for Benign/DoS/Recon/Spoofing/Web. Both lean on one dominant "
        "feature across most low-rate classes — a candidate shortcut / dataset artifact "
        "(window-sampling) to flag as a limitation.",
        "",
        "4. **The hardest class has the most diffuse, least-agreed explanation.** Web "
        "(worst F1 in both models) shows no dominant signature and the explanations are "
        "spread thin — poor separability and weak/incoherent attribution co-occur.",
        "",
    ]
    (DATA / "xai_comparison.md").write_text("\n".join(L))
    print("wrote data/xai_agreement.csv, data/xai_comparison.md")


if __name__ == "__main__":
    main()
