"""Class-level coupling between prediction stability and explanation stability.

Section 4.4 reports how closely per-class prediction preservation tracks per-class
top-5 attribution stability under perturbation. Script 28 persists only the
overlap-count correlation; this script derives every coupling figure the
manuscript quotes from the saved flow-level records, so each one traces to a
file: the Jaccard and count correlations, a flow-cluster bootstrap interval,
the leave-out check without the two volumetric classes, and the pooled
preserved-versus-flipped contrast. No model inference is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import CLASSES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop", nargs="*", default=["Mirai", "DDoS"])
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pearson(x, y) -> float:
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def exact_permutation_p(x, y) -> float:
    centered_x = np.asarray(x, dtype=np.float64) - np.mean(x)
    centered_y = np.asarray(y, dtype=np.float64) - np.mean(y)
    denominator = np.linalg.norm(centered_x) * np.linalg.norm(centered_y)
    observed = abs(float(centered_x @ centered_y / denominator))
    permutations = np.asarray(list(itertools.permutations(centered_y)))
    statistics = np.abs(permutations @ centered_x / denominator)
    return float(np.mean(statistics >= observed - 1e-12))


def class_means(records: pd.DataFrame, classes) -> pd.DataFrame:
    return (
        records.groupby("class")[["pred_preserved", "top5_jaccard", "top5_overlap_count", "spearman"]]
        .mean()
        .loc[classes]
    )


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_parquet(args.records)
    classes = [name for name in CLASSES if name in set(records["class"])]
    means = class_means(records, classes)
    r_jaccard = pearson(means["pred_preserved"], means["top5_jaccard"])
    r_count = pearson(means["pred_preserved"], means["top5_overlap_count"])
    p_all = exact_permutation_p(means["pred_preserved"], means["top5_jaccard"])

    # Cluster bootstrap: resample whole flows (all their perturbation levels)
    # with replacement within each class, then recompute the class-level r.
    rng = np.random.default_rng(args.seed)
    per_class_flows = {
        name: [group for _, group in records[records["class"] == name].groupby("SampleID")]
        for name in classes
    }
    boot = np.empty(args.resamples)
    for index in range(args.resamples):
        preserved, jaccard = [], []
        for name in classes:
            flows = per_class_flows[name]
            picks = rng.integers(0, len(flows), len(flows))
            sample = pd.concat([flows[i] for i in picks], ignore_index=True)
            preserved.append(sample["pred_preserved"].mean())
            jaccard.append(sample["top5_jaccard"].mean())
        boot[index] = pearson(preserved, jaccard)
    ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])

    kept = [name for name in classes if name not in set(args.drop)]
    r_kept = pearson(means.loc[kept, "pred_preserved"], means.loc[kept, "top5_jaccard"])
    p_kept = exact_permutation_p(means.loc[kept, "pred_preserved"], means.loc[kept, "top5_jaccard"])

    pooled = (
        records.groupby("pred_preserved")[["top5_jaccard", "spearman"]]
        .agg(["mean", "size"])
    )

    summary = {
        "n_records": int(len(records)),
        "n_flows": int(records["SampleID"].nunique()),
        "classes": classes,
        "pearson_r_preserved_vs_jaccard": r_jaccard,
        "pearson_r_preserved_vs_count": r_count,
        "exact_permutation_p_all_classes": p_all,
        "bootstrap_resamples": args.resamples,
        "bootstrap_seed": args.seed,
        "bootstrap_unit": "flow (SampleID), resampled within class",
        "bootstrap_ci95_jaccard": [float(ci_low), float(ci_high)],
        "bootstrap_nan_fraction": float(np.mean(np.isnan(boot))),
        "dropped_classes": list(args.drop),
        "pearson_r_without_dropped": r_kept,
        "exact_permutation_p_without_dropped": p_kept,
        "pooled_jaccard_preserved": float(pooled.loc[True, ("top5_jaccard", "mean")]),
        "pooled_jaccard_flipped": float(pooled.loc[False, ("top5_jaccard", "mean")]),
        "pooled_spearman_preserved": float(pooled.loc[True, ("spearman", "mean")]),
        "pooled_spearman_flipped": float(pooled.loc[False, ("spearman", "mean")]),
        "n_preserved": int(pooled.loc[True, ("top5_jaccard", "size")]),
        "n_flipped": int(pooled.loc[False, ("top5_jaccard", "size")]),
    }
    (output_dir / "robustness_coupling.json").write_text(json.dumps(summary, indent=2) + "\n")
    means.reset_index().to_csv(output_dir / "robustness_coupling_per_class.csv", index=False)

    lines = [
        "# Prediction-stability / explanation-stability coupling",
        "",
        f"Source: `{args.records.name}` ({summary['n_records']} records, "
        f"{summary['n_flows']} flows, {len(classes)} classes).",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Pearson r, preserved vs top-5 Jaccard | {r_jaccard:+.3f} |",
        f"| Pearson r, preserved vs top-5 overlap count | {r_count:+.3f} |",
        f"| Exact permutation p (8 classes, Jaccard) | {p_all:.4f} |",
        f"| Flow-cluster bootstrap 95% CI (Jaccard, {args.resamples:,} resamples) "
        f"| [{ci_low:+.3f}, {ci_high:+.3f}] |",
        f"| Pearson r without {', '.join(args.drop)} | {r_kept:+.3f} |",
        f"| Exact permutation p without {', '.join(args.drop)} | {p_kept:.4f} |",
        f"| Pooled top-5 Jaccard, preserved / flipped | "
        f"{summary['pooled_jaccard_preserved']:.3f} / {summary['pooled_jaccard_flipped']:.3f} |",
        f"| Pooled Spearman, preserved / flipped | "
        f"{summary['pooled_spearman_preserved']:.3f} / {summary['pooled_spearman_flipped']:.3f} |",
        "",
        "With eight class-level observations the correlation is descriptive.",
    ]
    (output_dir / "robustness_coupling.md").write_text("\n".join(lines) + "\n")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "inputs": {str(args.records): sha256(args.records)},
        "outputs": ["robustness_coupling.json", "robustness_coupling_per_class.csv", "robustness_coupling.md"],
    }
    (output_dir / "robustness_coupling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
