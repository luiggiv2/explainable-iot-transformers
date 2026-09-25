"""Is the windowing regime recoverable once `Number` is removed?

The `no_number` ablation removes the `Number` field (packets per window). If the
remaining 38 fields still encode the window size, the ablation does not remove
the collection artifact; it only removes one explicit copy of it. This probe
answers that directly, on the leakage-safe splits:

1. describes the distribution of `Number` per category on train and test and
   justifies the regime threshold (Number >= 50) by the gap between the two
   windowing modes;
2. fits a depth-3 decision tree on the no_number feature set (train) to
   predict the regime (Number >= 50) and evaluates it on test;
3. fits depth-limited regression trees (depth 3 and 6) on the same features to
   predict `Number` itself, reporting test R^2 and MAE;
4. checks the closed-form recovery Number = Tot sum / AVG on test.

Torch-free (scikit-learn only); seeded; runs in seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, mean_absolute_error, r2_score,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from pipeline_config import CLASSES, feature_pairs


ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "data" / "revision"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=REV / "splits")
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=REV / "results" / "window_probe")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(args.split_dir / "train.parquet")
    test = pd.read_parquet(args.split_dir / "test.parquet")

    # 1. distribution of Number
    dist_rows = []
    for split, frame in (("train", train), ("test", test)):
        for class_name in CLASSES:
            number = frame.loc[frame["Category"] == class_name, "Number"].astype(float)
            dist_rows.append({
                "split": split, "class": class_name, "n": len(number),
                "min": number.min(), "p01": number.quantile(0.01), "median": number.median(),
                "p99": number.quantile(0.99), "max": number.max(), "mean": number.mean(),
                "frac_ge_threshold": float((number >= args.threshold).mean()),
            })
        number = frame["Number"].astype(float)
        dist_rows.append({
            "split": split, "class": "_ALL", "n": len(number), "min": number.min(),
            "p01": number.quantile(0.01), "median": number.median(), "p99": number.quantile(0.99),
            "max": number.max(), "mean": number.mean(),
            "frac_ge_threshold": float((number >= args.threshold).mean()),
            "rows_between_15_and_85": int(((number > 15) & (number < 85)).sum()),
        })
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(out / "number_distribution.csv", index=False)

    columns = [column for column, _ in feature_pairs("no_number")]
    names = [name for _, name in feature_pairs("no_number")]
    x_train = train[columns].to_numpy(np.float64)
    x_test = test[columns].to_numpy(np.float64)
    regime_train = (train["Number"] >= args.threshold).to_numpy()
    regime_test = (test["Number"] >= args.threshold).to_numpy()
    volumetric = {"DDoS", "DoS", "Mirai"}
    category_regime_test = test["Category"].isin(volumetric).to_numpy()

    # 2. regime classifier
    clf = DecisionTreeClassifier(max_depth=3, random_state=args.seed).fit(x_train, regime_train)
    pred = clf.predict(x_test)
    regime = {
        "test_accuracy": float(accuracy_score(regime_test, pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(regime_test, pred)),
        "test_errors": int((pred != regime_test).sum()),
        "test_rows": int(len(pred)),
        "confusion_true_rows_pred_cols": confusion_matrix(regime_test, pred).tolist(),
        "accuracy_vs_category_regime": float(accuracy_score(category_regime_test, pred)),
        "number_regime_vs_category_regime_agreement": float(np.mean(regime_test == category_regime_test)),
        "rules": export_text(clf, feature_names=names, decimals=3),
    }

    # 3. regressors for Number itself
    regressors = {}
    for depth in (3, 6):
        reg = DecisionTreeRegressor(max_depth=depth, random_state=args.seed).fit(x_train, train["Number"].to_numpy(float))
        estimate = reg.predict(x_test)
        regressors[f"depth_{depth}"] = {
            "test_r2": float(r2_score(test["Number"], estimate)),
            "test_mae": float(mean_absolute_error(test["Number"], estimate)),
        }

    # 4. closed form
    nonzero = test["AVG"] != 0
    ratio = test.loc[nonzero, "Tot sum"] / test.loc[nonzero, "AVG"]
    closed = {
        "rows_with_avg_nonzero": int(nonzero.sum()),
        "rows_total": int(len(test)),
        "fraction_exact_within_1e-6": float(np.isclose(ratio, test.loc[nonzero, "Number"], rtol=1e-6).mean()),
        "r2": float(r2_score(test.loc[nonzero, "Number"], ratio)),
    }

    result = {"threshold": args.threshold, "seed": args.seed, "feature_set": "no_number",
              "n_features": len(columns), "regime_classifier_depth3": regime,
              "number_regressors": regressors, "closed_form_tot_sum_over_avg": closed}
    (out / "window_probe.json").write_text(json.dumps(result, indent=2) + "\n")

    all_rows = dist[dist["class"] == "_ALL"].set_index("split")
    lines = [
        "# Window-regime recoverability without `Number`",
        "",
        "## Distribution of `Number` (packets per window)",
        "",
        "| Split | Class | n | min | p01 | median | p99 | max | mean | share >= threshold |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dist.to_dict("records"):
        lines.append(f"| {row['split']} | {row['class']} | {row['n']} | {row['min']:g} | {row['p01']:g} | "
                     f"{row['median']:g} | {row['p99']:g} | {row['max']:g} | {row['mean']:.2f} | {row['frac_ge_threshold']:.4f} |")
    lines += [
        "",
        f"Threshold justification: `Number` is bimodal at 10 and 100; only "
        f"{int(all_rows.loc['train', 'rows_between_15_and_85'])} of {int(all_rows.loc['train', 'n']):,} train rows and "
        f"{int(all_rows.loc['test', 'rows_between_15_and_85'])} of {int(all_rows.loc['test', 'n']):,} test rows lie strictly "
        f"between 15 and 85, so any cut inside that gap, including {args.threshold:g}, yields the same regime labels "
        "up to those rows.",
        "",
        "## Depth-3 decision tree, regime (Number >= threshold) from the 38 no_number fields",
        "",
        f"- test accuracy {regime['test_accuracy']:.4f}, balanced accuracy {regime['test_balanced_accuracy']:.4f} "
        f"({regime['test_errors']} errors in {regime['test_rows']:,} rows);",
        f"- confusion (rows = true regime low/high, cols = predicted): {regime['confusion_true_rows_pred_cols']};",
        f"- regime defined by `Number` agrees with the category-level regime (DDoS/DoS/Mirai = 100-packet) on "
        f"{regime['number_regime_vs_category_regime_agreement']:.4f} of test rows.",
        "",
        "```",
        regime["rules"].rstrip(),
        "```",
        "",
        "## Depth-limited regression trees for `Number`",
        "",
        "| Depth | test R^2 | test MAE (packets) |",
        "|---:|---:|---:|",
    ]
    for key, value in regressors.items():
        lines.append(f"| {key.split('_')[1]} | {value['test_r2']:.4f} | {value['test_mae']:.3f} |")
    lines += [
        "",
        "## Closed form",
        "",
        f"`Tot sum / AVG` equals `Number` (relative tolerance 1e-6) on {closed['fraction_exact_within_1e-6']:.4f} of the "
        f"{closed['rows_with_avg_nonzero']:,} test rows with AVG != 0 (R^2 = {closed['r2']:.6f}).",
        "",
    ]
    summary_path = out / "window_probe.md"
    summary_path.write_text("\n".join(lines))
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/37_window_regime_probe.py",
        "seed": args.seed,
        "threshold": args.threshold,
        "inputs": {split: {"path": str((args.split_dir / f"{split}.parquet").resolve().relative_to(ROOT)),
                           "sha256": sha256(args.split_dir / f"{split}.parquet")} for split in ("train", "test")},
        "outputs": {p.name: sha256(p) for p in (out / "number_distribution.csv", out / "window_probe.json", summary_path)},
    }
    (out / "window_probe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
