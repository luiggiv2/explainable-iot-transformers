"""Build fixed, SampleID-aligned cohorts for the revised XAI analyses.

The main cohort contains rows correctly classified by both the representative
DistilBERT model and XGBoost. A second cohort contains rows correctly classified
by the representative original and no-Number DistilBERT models. Sampling is
stratified by true class and never uses confidence or test-set performance to
choose a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import CLASSES


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "revision"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-predictions",
        type=Path,
        default=DATA
        / "runs"
        / "distilbert"
        / "original"
        / "seed_123"
        / "test_predictions.parquet",
    )
    parser.add_argument(
        "--no-number-predictions",
        type=Path,
        default=DATA
        / "runs"
        / "distilbert"
        / "no_number"
        / "seed_42"
        / "test_predictions.parquet",
    )
    parser.add_argument(
        "--xgb-predictions",
        type=Path,
        default=DATA
        / "runs"
        / "baselines"
        / "original"
        / "test_predictions_xgb.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DATA / "xai" / "cohorts"
    )
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path, prediction_name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"row", "SampleID", "true_category", "y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    if not frame["SampleID"].is_unique:
        raise ValueError(f"duplicate SampleID values in {path}")
    return frame[list(required)].rename(columns={"y_pred": prediction_name})


def aligned_merge(original, comparison, comparison_prediction):
    merged = original.merge(
        comparison,
        on=["row", "SampleID", "true_category", "y_true"],
        validate="one_to_one",
    )
    if len(merged) != len(original) or len(merged) != len(comparison):
        raise ValueError(
            f"alignment lost rows for {comparison_prediction}: "
            f"{len(original)}, {len(comparison)}, {len(merged)}"
        )
    expected = np.asarray(CLASSES, dtype=object)[merged["y_true"].to_numpy()]
    if not np.array_equal(expected, merged["true_category"].to_numpy()):
        raise ValueError("integer and string class labels disagree")
    return merged


def select_cohort(frame, prediction_columns, max_per_class, seed, cohort_id):
    correct = np.ones(len(frame), dtype=bool)
    for column in prediction_columns:
        correct &= frame[column].to_numpy() == frame["y_true"].to_numpy()

    rows = []
    counts = []
    for class_index, class_name in enumerate(CLASSES):
        pool = frame.index[
            correct & (frame["true_category"] == class_name)
        ].to_numpy()
        take_count = min(len(pool), max_per_class)
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, cohort_id, class_index])
        )
        selected = (
            pool
            if len(pool) <= max_per_class
            else np.sort(rng.choice(pool, max_per_class, replace=False))
        )
        cohort = frame.loc[
            selected, ["row", "SampleID", "true_category", "y_true"]
        ].copy()
        cohort.insert(0, "cohort_order", np.arange(len(cohort)))
        rows.append(cohort)
        counts.append(
            {
                "class": class_name,
                "eligible": len(pool),
                "selected": take_count,
            }
        )
    selected = pd.concat(rows, ignore_index=True)
    selected["cohort_order"] = np.arange(len(selected))
    if not selected["SampleID"].is_unique:
        raise ValueError("cohort contains duplicate SampleID values")
    return selected, pd.DataFrame(counts)


def main():
    args = parse_args()
    if args.max_per_class <= 0:
        raise ValueError("--max-per-class must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = args.original_predictions.resolve()
    no_number_path = args.no_number_predictions.resolve()
    xgb_path = args.xgb_predictions.resolve()
    original = load(original_path, "pred_original")
    no_number = load(no_number_path, "pred_no_number")
    xgb = load(xgb_path, "pred_xgb")

    original_xgb = aligned_merge(original, xgb, "pred_xgb")
    original_no_number = aligned_merge(original, no_number, "pred_no_number")
    main_cohort, main_counts = select_cohort(
        original_xgb,
        ("pred_original", "pred_xgb"),
        args.max_per_class,
        args.seed,
        cohort_id=1,
    )
    ablation_cohort, ablation_counts = select_cohort(
        original_no_number,
        ("pred_original", "pred_no_number"),
        args.max_per_class,
        args.seed,
        cohort_id=2,
    )

    main_path = output_dir / "original_xgb.parquet"
    ablation_path = output_dir / "original_no_number.parquet"
    main_cohort.to_parquet(main_path, index=False)
    ablation_cohort.to_parquet(ablation_path, index=False)
    main_counts.to_csv(output_dir / "original_xgb_counts.csv", index=False)
    ablation_counts.to_csv(
        output_dir / "original_no_number_counts.csv", index=False
    )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": args.seed,
        "max_per_class": args.max_per_class,
        "checkpoint_selection": {
            "original": "seed 123, median validation macro-F1",
            "no_number": "seed 42, median validation macro-F1",
        },
        "inputs": {
            "original_predictions": {
                "path": str(original_path.relative_to(ROOT)),
                "sha256": sha256(original_path),
            },
            "no_number_predictions": {
                "path": str(no_number_path.relative_to(ROOT)),
                "sha256": sha256(no_number_path),
            },
            "xgb_predictions": {
                "path": str(xgb_path.relative_to(ROOT)),
                "sha256": sha256(xgb_path),
            },
        },
        "outputs": {
            "original_xgb": {
                "path": str(main_path.relative_to(ROOT)),
                "rows": len(main_cohort),
                "sha256": sha256(main_path),
            },
            "original_no_number": {
                "path": str(ablation_path.relative_to(ROOT)),
                "rows": len(ablation_cohort),
                "sha256": sha256(ablation_path),
            },
        },
    }
    (output_dir / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    lines = [
        "# Revised XAI cohorts",
        "",
        "Rows are selected from intersections of correct predictions and aligned "
        "one-to-one by `SampleID`. The same rows must be used by both methods in "
        "each comparison.",
        "",
        "## DistilBERT original vs. XGBoost",
        "",
        main_counts.to_markdown(index=False),
        "",
        "## DistilBERT original vs. no-`Number`",
        "",
        ablation_counts.to_markdown(index=False),
        "",
        "Checkpoint choice used validation performance only. Cohort sampling used "
        "true class and joint correctness, not confidence or attribution values.",
        "",
    ]
    (output_dir / "cohort_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote revised XAI cohorts to {output_dir}")


if __name__ == "__main__":
    main()
