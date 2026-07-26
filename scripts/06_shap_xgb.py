"""Phase 4 (cross-check) — SHAP on the XGBoost baseline.

Companion to the DistilBERT Layer-IG analysis: TreeExplainer gives exact SHAP
values for the gradient-boosted trees, so we can compare *what each model looks
at* per attack class (RQ2/RQ3). Same protocol as 05_captum_ig.py — correctly
classified test rows per class, attribution toward the true class, per-sample
L1-normalization — and the same short feature names, so the two importance
tables line up column-for-column.

Outputs (data/):
  shap_xgb_feature_importance.csv   long: class x feature -> signed_mean, abs_mean, rank
  shap_xgb_top_features.md          human-readable top-8 features per class
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
N_PER_CLASS = 120

# original CICIoT2023 column -> serialized short key (from 02_serialize.py),
# so SHAP features share names with the Captum output.
RENAME = {
    "Protocol Type": "proto_num", "Time_To_Live": "ttl", "Rate": "rate",
    "Header_Length": "header_len", "fin_flag_number": "fin",
    "syn_flag_number": "syn", "rst_flag_number": "rst", "psh_flag_number": "psh",
    "ack_flag_number": "ack", "ece_flag_number": "ece", "cwr_flag_number": "cwr",
    "ack_count": "ack_cnt", "syn_count": "syn_cnt", "fin_count": "fin_cnt",
    "rst_count": "rst_cnt", "HTTP": "http", "HTTPS": "https", "DNS": "dns",
    "Telnet": "telnet", "SMTP": "smtp", "SSH": "ssh", "IRC": "irc", "TCP": "tcp",
    "UDP": "udp", "DHCP": "dhcp", "ARP": "arp", "ICMP": "icmp", "IGMP": "igmp",
    "IPv": "ipv", "LLC": "llc", "Tot sum": "tot_sum", "Min": "min", "Max": "max",
    "AVG": "avg", "Std": "std", "Tot size": "tot_size", "IAT": "iat",
    "Number": "num", "Variance": "var",
}


def main():
    rng = np.random.default_rng(SEED)
    xgb = joblib.load(DATA / "models" / "xgb.joblib")

    df = pd.read_parquet(DATA / "splits" / "test.parquet").reset_index(drop=True)
    cols = [c for c in df.columns if c not in ("Label", "Category")]
    feat_names = [RENAME[c] for c in cols]
    X = df[cols].to_numpy(dtype=np.float32)
    y = pd.Categorical(df["Category"], categories=CLASSES).codes
    pred = xgb.predict(X)
    correct = pred == y
    print(f"xgb test acc {correct.mean():.4f}")

    explainer = shap.TreeExplainer(xgb)
    sv = explainer.shap_values(X)                 # multiclass: (n, f, k) or list of k
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
    print(f"shap values shape {sv.shape}")         # expect (n_samples, n_features, n_classes)

    agg_rows = []
    for c, cls in enumerate(CLASSES):
        pool = np.where((y == c) & correct)[0]
        take = pool if len(pool) <= N_PER_CLASS else rng.choice(pool, N_PER_CLASS, replace=False)
        vals = sv[take, :, c]                       # (m, f) SHAP toward class c
        l1 = np.abs(vals).sum(axis=1, keepdims=True)
        vals = vals / np.where(l1 == 0, 1.0, l1)    # per-sample L1-normalize
        signed = vals.mean(axis=0)
        absmean = np.abs(vals).mean(axis=0)
        order = np.argsort(absmean)[::-1]
        for rank, fi in enumerate(order, 1):
            agg_rows.append({"class": cls, "feature": feat_names[fi],
                             "signed_mean": round(float(signed[fi]), 6),
                             "abs_mean": round(float(absmean[fi]), 6),
                             "rank": rank})
        print(f"  {cls}: {len(take)} samples")

    agg = pd.DataFrame(agg_rows)
    agg.to_csv(DATA / "shap_xgb_feature_importance.csv", index=False)

    lines = ["# XGBoost — SHAP (top features per class)", "",
             f"TreeExplainer; {N_PER_CLASS} correctly-classified test samples/class; "
             "SHAP toward the true class, L1-normalized per sample (same protocol as "
             "the DistilBERT Layer-IG analysis). signed_mean>0 pushes toward the class.", ""]
    for cls in CLASSES:
        lines.append(f"## {cls}")
        top = agg[agg["class"] == cls].nsmallest(8, "rank")
        for _, r in top.iterrows():
            arrow = "+" if r["signed_mean"] >= 0 else "-"
            lines.append(f"- `{r['feature']}`  |shap|={r['abs_mean']:.4f}  ({arrow}{abs(r['signed_mean']):.4f})")
        lines.append("")
    (DATA / "shap_xgb_top_features.md").write_text("\n".join(lines))
    print("wrote shap_xgb_feature_importance.csv, shap_xgb_top_features.md")


if __name__ == "__main__":
    main()
