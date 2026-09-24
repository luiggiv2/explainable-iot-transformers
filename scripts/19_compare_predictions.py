"""Paired statistical comparison using aligned, run-owned prediction records."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from pipeline_config import CLASSES


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distilbert-predictions",
        type=Path,
        required=True,
        help="test_predictions.parquet emitted by scripts/04_finetune.py.",
    )
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=DATA / "revision" / "runs" / "baselines" / "original",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_predictions(path: Path, model: str):
    frame = pd.read_parquet(path)
    required = {"SampleID", "y_true", "y_pred", "true_category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    if not frame["SampleID"].is_unique:
        raise ValueError(f"duplicate SampleID values in {path}")
    return frame[list(required)].rename(columns={"y_pred": f"pred_{model}"})


def mcnemar(a_correct, b_correct):
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    discordant = a_only + b_only
    p_value = (
        binomtest(min(a_only, b_only), discordant, 0.5).pvalue
        if discordant
        else 1.0
    )
    return a_only, b_only, p_value


def main():
    args = parse_args()
    db = load_predictions(args.distilbert_predictions.resolve(), "distilbert")
    models = {"DistilBERT": "pred_distilbert"}
    merged = db
    for filename, display, key in (
        ("test_predictions_xgb.parquet", "XGBoost", "xgb"),
        ("test_predictions_rf.parquet", "Random Forest", "rf"),
    ):
        frame = load_predictions(args.baseline_run_dir.resolve() / filename, key)
        merged = merged.merge(
            frame,
            on=["SampleID", "y_true", "true_category"],
            validate="one_to_one",
        )
        models[display] = f"pred_{key}"
    if len(merged) != len(db):
        raise ValueError(
            f"prediction alignment lost rows: DistilBERT {len(db)}, joined {len(merged)}"
        )
    expected_category = np.asarray(CLASSES, dtype=object)[merged["y_true"].to_numpy()]
    if not np.array_equal(expected_category, merged["true_category"].to_numpy()):
        raise ValueError("integer and string test labels disagree")

    y = merged["y_true"].to_numpy()
    point = {
        model: f1_score(y, merged[column], average="macro")
        for model, column in models.items()
    }
    rng = np.random.default_rng(args.seed)
    n = len(merged)
    boot = {model: np.empty(args.bootstrap_resamples) for model in models}
    differences = {
        model: np.empty(args.bootstrap_resamples)
        for model in models
        if model != "DistilBERT"
    }
    for iteration in range(args.bootstrap_resamples):
        indices = rng.integers(0, n, n)
        y_boot = y[indices]
        values = {
            model: f1_score(
                y_boot,
                merged[column].to_numpy()[indices],
                average="macro",
            )
            for model, column in models.items()
        }
        for model, value in values.items():
            boot[model][iteration] = value
        for model in differences:
            differences[model][iteration] = values["DistilBERT"] - values[model]

    lines = [
        "# Paired model comparison on the leakage-safe test split",
        "",
        f"All predictions were joined one-to-one by `SampleID` (n={n}). DistilBERT "
        "predictions come from the same FP32 run artifact as its reported metrics.",
        "",
        "## Macro-F1",
        "",
        "| Model | point estimate | bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for model in models:
        lower, upper = np.percentile(boot[model], [2.5, 97.5])
        lines.append(f"| {model} | {point[model]:.4f} | [{lower:.4f}, {upper:.4f}] |")
    lines.extend(
        [
            "",
            "## Paired macro-F1 differences",
            "",
            "| Comparison | point difference | bootstrap 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for model, distribution in differences.items():
        lower, upper = np.percentile(distribution, [2.5, 97.5])
        lines.append(
            f"| DistilBERT - {model} | {point['DistilBERT'] - point[model]:+.4f} | "
            f"[{lower:+.4f}, {upper:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Exact McNemar tests",
            "",
            "| Comparison | DB only correct | baseline only correct | exact p |",
            "|---|---:|---:|---:|",
        ]
    )
    db_correct = merged[models["DistilBERT"]].to_numpy() == y
    for model in ("XGBoost", "Random Forest"):
        baseline_correct = merged[models[model]].to_numpy() == y
        db_only, baseline_only, p_value = mcnemar(db_correct, baseline_correct)
        lines.append(
            f"| DistilBERT vs {model} | {db_only} | {baseline_only} | {p_value:.3e} |"
        )
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
