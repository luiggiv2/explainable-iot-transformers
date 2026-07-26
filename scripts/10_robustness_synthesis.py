"""Phase 5 synthesis — the honest RQ4 findings.

The all-39-feature Spearman was flat (~0.47) across noise because it is dominated
by the near-zero attribution tail (the eps=0 control gives a perfect 1.000 floor,
so IG itself is deterministic — the tail reshuffles the instant the input moves).
The meaningful metrics are top-5 feature overlap and prediction stability, and
they track each other. This script recomputes those from the saved per-sample
records and writes the findings artifact the manuscript draws on.

Inputs:  data/robustness_records_partial.csv (per-sample), data/robustness_metrics.csv
Output:  data/robustness_findings.md
Note:    IG noise floor (eps=0, identical input attributed twice) = 1.000 for all
         classes over both all-39 and top-8 features (scripts/09_ig_noise_floor.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
EPS = [0.01, 0.05, 0.10]


def main():
    rec = pd.read_csv(DATA / "robustness_records_partial.csv")
    met = pd.read_csv(DATA / "robustness_metrics.csv")

    per = []
    for c in CLASSES:
        sub = rec[rec["class"] == c]
        per.append({"class": c, "jaccard5": sub["jaccard5"].mean(),
                    "pred_preserved": sub["pred_preserved"].mean()})
    per = pd.DataFrame(per)
    r, _ = pearsonr(per["pred_preserved"], per["jaccard5"])
    pres = rec[rec["pred_preserved"]]["jaccard5"].mean()
    flip = rec[~rec["pred_preserved"]]["jaccard5"].mean()

    L = ["# Explanation robustness (RQ4) — findings", "",
         "DistilBERT Layer-IG, 60 correctly-classified flows/class, multiplicative "
         "Gaussian noise at 1/5/10%, re-attributed toward the true class. Sources: "
         "robustness_metrics.csv, robustness_records_partial.csv, "
         "09_ig_noise_floor.py.", "",
         "## Method validity — IG is deterministic",
         "eps=0 control (identical input attributed twice): Spearman = **1.000** for "
         "all classes over both all-39 and top-8 features. Any instability below is "
         "genuine input sensitivity, not method noise.", "",
         "## Metric caveat",
         "Spearman over all 39 features is flat at ~0.47 across noise levels because "
         "it is dominated by the ~30 near-zero-attribution features, whose order "
         "reshuffles the instant the input moves and then saturates. We therefore "
         "report **top-5 feature overlap** and **prediction stability** as the "
         "headline metrics.", "",
         "## Explanation and decision stability track each other", "",
         "| Class | pred preserved | top-5 Jaccard |",
         "|---|---|---|"]
    for _, x in per.sort_values("pred_preserved", ascending=False).iterrows():
        L.append(f"| {x['class']} | {x['pred_preserved']:.1%} | {x['jaccard5']:.3f} |")
    L += ["",
          f"Cross-class correlation between prediction stability and top-5 explanation "
          f"stability: **r = {r:+.3f}**. Top-5 Jaccard is {pres:.3f} when the prediction "
          f"is preserved vs {flip:.3f} when it flips.", "",
          "## Reading",
          "- **Robust where confident:** the well-separated volumetric classes (Mirai "
          "~98% predictions preserved, DDoS ~88%) keep both decision and top features "
          "under noise.",
          "- **Fragile where confused:** the low-rate classes the model already "
          "confuses (BruteForce ~2%, Recon ~5% predictions preserved) flip class under "
          "even 1% noise and their explanations reshuffle — decision and explanation "
          "fragility are concentrated in the same classes (ties RQ4 back to RQ1/RQ3).",
          "- Even at best, top-5 stability is moderate (~0.45 for Mirai ≈ 3/5 features "
          "retained), so explanations should be read at the class-population level, not "
          "trusted per-individual-flow for the low-rate categories.", ""]
    (DATA / "robustness_findings.md").write_text("\n".join(L))
    print(f"pred<->explanation r={r:+.3f}; preserved Jacc={pres:.3f} flip={flip:.3f}")
    print("wrote data/robustness_findings.md")


if __name__ == "__main__":
    main()
