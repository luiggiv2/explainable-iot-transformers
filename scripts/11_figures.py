"""Phase 6 — publication figures.

Static paper figures (PDF vector + PNG 300dpi) from the data/ artifacts.
Color: cividis for sequential magnitude (perceptually uniform, CVD-safe,
grayscale-degradable), Okabe-Ito for categorical identity (CVD-safe). Recessive
axes, direct labels, per-cell annotation only where it stays legible.

  fig1_perclass_f1        models x classes F1 heatmap + macro-F1 (RQ1)
  fig2_confusion          DistilBERT row-normalized confusion matrix
  fig3_feature_importance DistilBERT-IG vs XGBoost-SHAP per-class importance (RQ2/RQ3)
  fig4_robustness         prediction vs explanation stability per class (RQ4)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = DATA / "figures"
CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
OKABE = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
         "orange": "#E69F00", "purple": "#CC79A7", "sky": "#56B4E9"}
INK, MUTED = "#222222", "#666666"

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  {name}.pdf/.png")


def f1_matrix():
    b = pd.read_csv(DATA / "baselines_metrics.csv")
    t = pd.read_csv(DATA / "transformer_metrics.csv")
    allm = pd.concat([b, t])
    models = [("xgb", "XGBoost"), ("rf", "RF"), ("distilbert", "DistilBERT"), ("svm", "SVM")]
    M = np.array([[allm[(allm.model == m) & (allm["class"] == c)].f1.iloc[0]
                   for c in CLASSES] for m, _ in models])
    macro = np.array([allm[(allm.model == m) & (allm["class"] == "_macro_avg")].f1.iloc[0]
                      for m, _ in models])
    return M, macro, [n for _, n in models]


def fig1():
    M, macro, names = f1_matrix()
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    im = ax.imshow(M, cmap="cividis", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v < 0.72 else INK)
    # macro-F1 as a right-side annotation column
    for i, mv in enumerate(macro):
        ax.text(len(CLASSES) - 0.3, i, f"  macro {mv:.3f}", ha="left", va="center",
                fontsize=7.5, color=MUTED)
    ax.set_title("Per-class F1 by model (CICIoT2023 test)", fontsize=10, color=INK, pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.16)
    cb.set_label("F1", color=MUTED); cb.ax.tick_params(labelsize=7)
    save(fig, "fig1_perclass_f1")


def fig2():
    cm = pd.read_csv(DATA / "confusion_distilbert.csv", index_col=0).to_numpy(float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    im = ax.imshow(cmn, cmap="cividis", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(8), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(8), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(8):
        for j in range(8):
            if cmn[i, j] >= 0.01:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cmn[i, j] < 0.6 else INK)
    ax.set_title("DistilBERT confusion (row-normalized)", fontsize=10, color=INK, pad=8)
    save(fig, "fig2_confusion")


def fig3():
    ig = pd.read_csv(DATA / "captum_feature_attribution.csv")
    sh = pd.read_csv(DATA / "shap_xgb_feature_importance.csv")
    # column set = union of each model's top-5 per class, ordered by serialization grouping
    order = ["proto_num", "ttl", "rate", "header_len", "syn", "psh", "ack", "syn_cnt",
             "ssh", "https", "icmp", "tot_sum", "min", "max", "avg", "std",
             "tot_size", "iat", "num", "var"]
    top = set()
    for df in (ig, sh):
        for c in CLASSES:
            top |= set(df[df["class"] == c].nsmallest(5, "rank")["feature"])
    feats = [f for f in order if f in top]

    def mat(df):
        return np.array([[df[(df["class"] == c) & (df.feature == f)]["abs_mean"].iloc[0]
                          for f in feats] for c in CLASSES])
    A, B = mat(ig), mat(sh)
    A = A / A.max(axis=1, keepdims=True)      # row-normalize within each panel
    B = B / B.max(axis=1, keepdims=True)      # so each class's profile is legible
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.8))
    for ax, Mx, title in [(axes[0], A, "DistilBERT — Layer Integrated Gradients"),
                          (axes[1], B, "XGBoost — SHAP")]:
        im = ax.imshow(Mx, cmap="cividis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(feats)), feats, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(8), CLASSES, fontsize=8)
        ax.set_title(title, fontsize=9.5, color=INK, pad=5)
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label("rel. importance", fontsize=6.5, color=MUTED)
    fig.suptitle("Per-class feature importance: what each model attends to",
                 fontsize=10.5, color=INK, y=0.99)
    fig.text(0.5, 0.945, "each row normalized to its maximum", ha="center",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig3_feature_importance")


def fig4():
    rec = pd.read_csv(DATA / "robustness_records_partial.csv")
    per = rec.groupby("class").agg(pred=("pred_preserved", "mean"),
                                   jac=("jaccard5", "mean")).reindex(CLASSES)
    volumetric = {"DDoS", "DoS", "Mirai"}
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    x, yv = per["pred"].to_numpy(), per["jac"].to_numpy()
    # regression line
    m, b = np.polyfit(x, yv, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, m * xs + b, color=MUTED, lw=1.2, ls="--", zorder=1)
    r, _ = pearsonr(x, yv)
    for c in CLASSES:
        px, py = per.loc[c, "pred"], per.loc[c, "jac"]
        col = OKABE["blue"] if c in volumetric else OKABE["vermillion"]
        ax.scatter(px, py, s=70, color=col, zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(c, (px, py), textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color=INK)
    ax.scatter([], [], color=OKABE["blue"], label="volumetric (DDoS/DoS/Mirai)")
    ax.scatter([], [], color=OKABE["vermillion"], label="low-rate")
    ax.set_xlabel("Prediction preserved under noise")
    ax.set_ylabel("Explanation stability (top-5 Jaccard)")
    ax.set_title(f"Explanation stability tracks decision stability (r = {r:+.2f})",
                 fontsize=9.5, color=INK, pad=8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(True, color="#eee", lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig4_robustness")


def main():
    FIG.mkdir(exist_ok=True)
    print("writing figures to data/figures/:")
    fig1(); fig2(); fig3(); fig4()
    print("done")


if __name__ == "__main__":
    main()
