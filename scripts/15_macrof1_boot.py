"""Phase B (review) — macro-F1 difference bootstrap + RF seed variance. [torch-free]

Directly tests the RQ1 gap on the exact reported metric (macro-F1), complementing
the accuracy-based McNemar. Bootstraps the shared test set, recomputes macro-F1
for DistilBERT / XGBoost / RF each resample, and reports 95% CIs on each model's
macro-F1 and on the paired differences. Also refits Random Forest under several
seeds to quantify baseline run-to-run variance. Imports NO torch.

Output: data/macrof1_boot.md
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
rng = np.random.default_rng(SEED)


def load_xy(split):
    df = pd.read_parquet(DATA / "splits" / f"{split}.parquet")
    X = df.drop(columns=["Label", "Category"]).to_numpy(np.float32)
    y = pd.Categorical(df["Category"], categories=CLASSES).codes
    return X, np.asarray(y)


def main():
    Xte, y = load_xy("test")
    db = np.load(DATA / "distilbert_test_preds.npy")
    xgb = joblib.load(DATA / "models" / "xgb.joblib"); xgb.n_jobs = 1
    rf = joblib.load(DATA / "models" / "rf.joblib"); rf.n_jobs = 1
    xg = xgb.predict(Xte)
    rfp = rf.predict(Xte)
    preds = {"DistilBERT": db, "XGBoost": xg, "RF": rfp}
    macro = {k: f1_score(y, p, average="macro") for k, p in preds.items()}

    B, n = 4000, len(y)
    dist = {k: np.empty(B) for k in preds}
    d_xgb = np.empty(B); d_rf = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        mf = {k: f1_score(yb, p[idx], average="macro") for k, p in preds.items()}
        for k in preds:
            dist[k][b] = mf[k]
        d_xgb[b] = mf["DistilBERT"] - mf["XGBoost"]
        d_rf[b] = mf["DistilBERT"] - mf["RF"]

    L = ["# Macro-F1 bootstrap and baseline seed variance (reviewer #2)", "",
         f"Test-set bootstrap ({B} resamples, n={n}).", "",
         "## Per-model macro-F1 (point estimate and bootstrap 95% CI)"]
    for k in ["XGBoost", "RF", "DistilBERT"]:
        lo, hi = np.percentile(dist[k], [2.5, 97.5])
        L.append(f"- {k}: {macro[k]:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    L += ["", "## Paired macro-F1 difference (DistilBERT − baseline)"]
    for name, d in [("XGBoost", d_xgb), ("RF", d_rf)]:
        lo, hi = np.percentile(d, [2.5, 97.5])
        pgt = (np.sum(d >= 0) + 1) / (B + 1)   # P(DistilBERT >= baseline)
        excl = "excludes 0 (significant)" if hi < 0 or lo > 0 else "includes 0 (not significant)"
        L.append(f"- DistilBERT − {name}: {macro['DistilBERT']-macro[name]:+.4f}  "
                 f"95% CI [{lo:+.4f}, {hi:+.4f}] → {excl}")
    L += [""]

    # RF seed variance (refit; XGBoost/SVM are effectively deterministic given fixed
    # data with no subsampling, so seed variance is negligible and not refit here)
    Xtr, ytr = load_xy("train")
    accs = []
    for s in (1, 7, 21, 42, 123):
        m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=s)
        m.fit(Xtr, ytr)
        accs.append(f1_score(y, m.predict(Xte), average="macro"))
    L += ["## Random Forest macro-F1 across 5 seeds (refit)",
          f"- seeds {[1,7,21,42,123]}: " + ", ".join(f"{a:.4f}" for a in accs),
          f"- mean {np.mean(accs):.4f}, std {np.std(accs):.4f} "
          "(XGBoost and SVM are deterministic given fixed data / no subsampling, so their "
          "seed variance is negligible; DistilBERT multi-seed variance is left to future "
          "work per the compute budget).", ""]

    (DATA / "macrof1_boot.md").write_text("\n".join(L))
    print("\n".join(L))
    print("wrote data/macrof1_boot.md")


if __name__ == "__main__":
    main()
