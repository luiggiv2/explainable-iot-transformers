"""Phase B (review response) — statistical hardening.  [torch-free]

The first versions crashed natively on macOS because importing torch loads a
second OpenMP runtime that conflicts with XGBoost/sklearn in the same process.
DistilBERT test predictions are precomputed once (scripts/13_make_preds.py ->
data/distilbert_test_preds.npy); this script imports NO torch/transformers and
does only pure numpy/pandas/joblib/scipy work.

  - McNemar exact test: DistilBERT vs XGBoost and vs RF on the shared test split.
  - RQ4 coupling r: Pearson over 8 classes + sample-level bootstrap 95% CI +
    permutation p + leave-two-volumetric-out sensitivity.
  - RQ4 prediction-preservation monotonicity across noise levels.
  - RQ2/RQ3 IG top-feature ranking stability (bootstrap over samples).

Output: data/stats_hardening.md
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
PREDS = DATA / "distilbert_test_preds.npy"
rng = np.random.default_rng(SEED)
OUT = DATA / "stats_hardening.md"
_buf = []


def emit(s=""):
    _buf.append(s)
    OUT.write_text("\n".join(_buf))
    print(s, flush=True)


def sk_preds(name):
    m = joblib.load(DATA / "models" / f"{name}.joblib")
    if hasattr(m, "n_jobs"):
        m.n_jobs = 1
    df = pd.read_parquet(DATA / "splits" / "test.parquet")
    X = df.drop(columns=["Label", "Category"]).to_numpy(np.float32)
    y = pd.Categorical(df["Category"], categories=CLASSES).codes
    return m.predict(X), np.asarray(y)


def mcnemar(a_correct, b_correct):
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
    return b, c, p


def main():
    if not PREDS.exists():
        sys.exit("Missing data/distilbert_test_preds.npy — run scripts/13_make_preds.py first")
    db = np.load(PREDS)
    emit("# Statistical hardening (reviewer Phase B)")
    emit("")

    # ---- McNemar
    print("[stage] McNemar", flush=True)
    xg, y = sk_preds("xgb")
    rf, y2 = sk_preds("rf")
    assert np.array_equal(y, y2) and len(db) == len(y), "rows misaligned"
    db_c, xg_c, rf_c = db == y, xg == y, rf == y
    emit("## McNemar exact test (paired, shared n=9000 test rows)")
    emit(f"- accuracy: DistilBERT {db_c.mean():.4f}, XGBoost {xg_c.mean():.4f}, RF {rf_c.mean():.4f}")
    for name, oc in [("XGBoost", xg_c), ("RF", rf_c)]:
        b, c, p = mcnemar(db_c, oc)
        sig = "significant" if p < 0.05 else "NOT significant"
        emit(f"- DistilBERT vs {name}: discordant b(DB right, {name} wrong)={b}, "
             f"c(DB wrong, {name} right)={c}; exact McNemar p={p:.3e} → {sig} at alpha=0.05.")
    emit("")

    # ---- RQ4 coupling r
    print("[stage] RQ4 coupling", flush=True)
    rec = pd.read_csv(DATA / "robustness_records_partial.csv")
    P = {c: rec[rec["class"] == c]["pred_preserved"].to_numpy(float) for c in CLASSES}
    J = {c: rec[rec["class"] == c]["jaccard5"].to_numpy(float) for c in CLASSES}
    p_obs = np.array([P[c].mean() for c in CLASSES])
    j_obs = np.array([J[c].mean() for c in CLASSES])
    r_obs = pearsonr(p_obs, j_obs)[0]
    B = 3000
    boots = np.empty(B)
    for k in range(B):
        pv = np.array([P[c][rng.integers(0, len(P[c]), len(P[c]))].mean() for c in CLASSES])
        jv = np.array([J[c][rng.integers(0, len(J[c]), len(J[c]))].mean() for c in CLASSES])
        boots[k] = pearsonr(pv, jv)[0] if pv.std() and jv.std() else np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    perm = np.array([abs(pearsonr(p_obs, rng.permutation(j_obs))[0]) for _ in range(20000)])
    p_perm = (np.sum(perm >= abs(r_obs)) + 1) / (len(perm) + 1)
    keep = [i for i, c in enumerate(CLASSES) if c not in ("Mirai", "DDoS")]
    r_l2o, p_l2o = pearsonr(p_obs[keep], j_obs[keep])
    emit("## RQ4 coupling (prediction-preserved vs top-5 Jaccard, n=8 classes)")
    emit(f"- Pearson r = {r_obs:+.3f}; sample-level bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}] ({B} resamples)")
    emit(f"- permutation p = {p_perm:.4f} (20000 permutations)")
    emit(f"- leave-two-out (drop Mirai, DDoS): r = {r_l2o:+.3f}, p = {p_l2o:.3f} on 6 classes")
    emit("")

    # ---- RQ4 non-monotonicity
    print("[stage] monotonicity", flush=True)
    met = pd.read_csv(DATA / "robustness_metrics.csv")
    eps = sorted(met["eps"].unique())
    emit("## RQ4 prediction-preservation across noise levels")
    emit("| Class | " + " | ".join(f"{e:.0%}" for e in eps) + " | monotone non-increasing? |")
    emit("|---|" + "---|" * (len(eps) + 1))
    nonmono = 0
    for c in CLASSES:
        v = [met[(met["class"] == c) & (met["eps"] == e)]["pred_preserved_frac"].iloc[0] for e in eps]
        mono = all(v[i] >= v[i+1] for i in range(len(v)-1))
        nonmono += (not mono)
        emit(f"| {c} | " + " | ".join(f"{x:.1%}" for x in v) + f" | {'yes' if mono else '**NO**'} |")
    emit("")
    emit(f"{nonmono}/8 classes are non-monotone: prediction preservation does not fall "
         "monotonically as noise grows, indicating the levels do not behave as a graded "
         "robustness sweep (small per-class n).")
    emit("")

    # ---- RQ2/RQ3 ranking stability
    print("[stage] ranking bootstrap", flush=True)
    ps = pd.read_parquet(DATA / "captum_per_sample.parquet")
    feat = [c for c in ps.columns if c not in ("class", "row")]
    emit("## RQ2/RQ3 IG ranking stability (1000-sample bootstrap per class)")
    emit("| Class | observed top-1 | P(stays top-1) | P(top-1 stays in top-3) |")
    emit("|---|---|---|---|")
    for c in CLASSES:
        M = ps[ps["class"] == c][feat].abs().to_numpy()
        top1 = int(np.argsort(M.mean(0))[::-1][0])
        s1 = s3 = 0
        for _ in range(1000):
            idx = rng.integers(0, len(M), len(M))
            r = np.argsort(M[idx].mean(0))[::-1]
            s1 += (r[0] == top1); s3 += (top1 in set(r[:3]))
        emit(f"| {c} | `{feat[top1]}` | {s1/1000:.2f} | {s3/1000:.2f} |")
    emit("")
    print("wrote data/stats_hardening.md", flush=True)


if __name__ == "__main__":
    main()
