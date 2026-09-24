"""Aggregate completed DistilBERT seeds without selecting on test performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import CLASSES


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEEDS = (42, 123, 2026)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-set",
        choices=("original", "no_number"),
        default="original",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DATA / "revision" / "results"
    )
    return parser.parse_args()


def aggregate(frame: pd.DataFrame, columns):
    rows = []
    for column in columns:
        values = frame[column]
        rows.append(
            {
                "metric": column,
                "mean": values.mean(),
                "sample_sd": values.std(ddof=1),
                "min": values.min(),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    run_root = (
        args.run_root
        or DATA / "revision" / "runs" / "distilbert" / args.feature_set
    ).resolve()
    baseline_metrics = (
        args.baseline_metrics
        or DATA
        / "revision"
        / "runs"
        / "baselines"
        / args.feature_set
        / "baselines_metrics.csv"
    ).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = []
    per_class = []
    fixed_reference = None
    fixed_fields = (
        "model_name",
        "feature_set",
        "classes",
        "serialized_dir",
        "train_precision",
        "validation_precision",
        "test_precision",
        "device",
        "max_length",
        "batch_size",
        "eval_batch_size",
        "learning_rate",
        "max_epochs",
        "patience",
        "warmup_fraction",
        "rows",
    )
    for seed in SEEDS:
        run_dir = run_root / f"seed_{seed}"
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        if manifest["seed"] != seed:
            raise ValueError(f"seed mismatch in {run_dir}")
        if manifest["feature_set"] != args.feature_set:
            raise ValueError(
                f"feature-set mismatch in {run_dir}: "
                f"expected {args.feature_set}, found {manifest['feature_set']}"
            )
        if not all(manifest["consistency_checks"].values()):
            raise ValueError(f"failed consistency check in {run_dir}")
        fixed = {field: manifest[field] for field in fixed_fields}
        fixed["serialized_test_sha256"] = manifest["sha256"]["serialized_test"]
        if fixed_reference is None:
            fixed_reference = fixed
        elif fixed != fixed_reference:
            raise ValueError(f"protocol drift detected in seed {seed}")

        metrics = pd.read_csv(run_dir / "transformer_metrics.csv")
        lookup = metrics.set_index("class")["f1"]
        manifests.append(
            {
                "seed": seed,
                "best_epoch": manifest["best_epoch"],
                "validation_macro_f1": manifest["best_val_macro_f1_fp32"],
                "test_macro_f1": manifest["test_macro_f1_fp32"],
                "test_accuracy": manifest["test_accuracy_fp32"],
                "test_weighted_f1": lookup["_weighted_avg"],
                "test_roc_auc_macro": lookup["_roc_auc_macro_ovr"],
                "train_time_hours": lookup["_train_time_s"] / 3600,
            }
        )
        for cls in CLASSES:
            row = metrics[metrics["class"] == cls].iloc[0]
            per_class.append(
                {
                    "seed": seed,
                    "class": cls,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                }
            )

    runs = pd.DataFrame(manifests)
    class_runs = pd.DataFrame(per_class)
    summary = aggregate(
        runs,
        (
            "validation_macro_f1",
            "test_macro_f1",
            "test_accuracy",
            "test_weighted_f1",
            "test_roc_auc_macro",
            "train_time_hours",
        ),
    )
    class_summary = (
        class_runs.groupby("class")[["precision", "recall", "f1"]]
        .agg(["mean", "std", "min", "max"])
        .reindex(CLASSES)
    )
    class_summary.columns = ["_".join(column) for column in class_summary.columns]
    class_summary = class_summary.reset_index()

    baseline = pd.read_csv(baseline_metrics)
    baseline_macro = (
        baseline[baseline["class"] == "_macro_avg"]
        .set_index("model")["f1"]
        .to_dict()
    )
    comparison_rows = []
    for model in ("xgb", "rf"):
        differences = runs["test_macro_f1"] - baseline_macro[model]
        comparison_rows.append(
            {
                "baseline": model,
                "baseline_macro_f1": baseline_macro[model],
                "mean_distilbert_minus_baseline": differences.mean(),
                "sample_sd_across_seeds": differences.std(ddof=1),
                "min_difference": differences.min(),
                "max_difference": differences.max(),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)

    # Predeclared representative checkpoint: median validation performance.
    ordered = runs.sort_values(["validation_macro_f1", "seed"])
    representative = int(ordered.iloc[len(ordered) // 2]["seed"])

    prefix = f"distilbert_{args.feature_set}"
    runs.to_csv(output_dir / f"{prefix}_seeds.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_multiseed_summary.csv", index=False)
    class_summary.to_csv(
        output_dir / f"{prefix}_multiseed_per_class.csv", index=False
    )
    comparisons.to_csv(
        output_dir / f"{prefix}_baseline_comparison.csv", index=False
    )

    display_runs = runs.copy()
    display_runs["train_time_hours"] = display_runs["train_time_hours"].round(2)
    lines = [
        f"# DistilBERT `{args.feature_set}` feature set — three-seed summary",
        "",
        "Seeds 42, 123, and 2026 use an identical leakage-safe split and fixed "
        "training protocol. Values below describe training stochasticity; n=3 is too "
        "small for strong distributional inference.",
        "",
        "## Per-seed results",
        "",
        display_runs.round(4).to_markdown(index=False),
        "",
        "## Aggregate across seeds",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "## Difference from fixed classical baselines",
        "",
        comparisons.round(4).to_markdown(index=False),
        "",
        "## Per-class F1 across seeds",
        "",
        class_summary[["class", "f1_mean", "f1_std", "f1_min", "f1_max"]]
        .round(4)
        .to_markdown(index=False),
        "",
        f"The predeclared median-validation checkpoint for full XAI is **seed "
        f"{representative}**. This selection does not inspect test performance.",
        "",
        "Test-set bootstrap and across-seed variability quantify different sources "
        "of uncertainty and must be reported separately.",
        "",
    ]
    (output_dir / f"{prefix}_multiseed_summary.md").write_text(
        "\n".join(lines)
    )
    print("\n".join(lines))
    print(f"wrote multiseed artifacts to {output_dir}")


if __name__ == "__main__":
    main()
