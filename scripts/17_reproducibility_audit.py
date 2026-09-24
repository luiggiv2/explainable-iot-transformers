"""Freeze and audit an experiment state without importing torch or XGBoost.

The manifest records checksums for the split, serialization, predictions,
reported metrics, and checkpoint. The Markdown report checks row alignment,
cross-split duplicates, ambiguous identical windows, and whether saved
DistilBERT predictions reconstruct the published confusion matrix.

This script is intentionally read-only with respect to model artifacts. Its
only outputs are the JSON manifest and Markdown audit selected on the CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from pipeline_config import (
    CLASSES,
    FEATURE_COLUMNS,
    FEATURE_SET_EXCLUSIONS,
    present_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_DATA / "baseline_manifest.json"
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_DATA / "baseline_audit.md"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--distilbert-run-dir",
        type=Path,
        default=None,
        help="Run-owned predictions, metrics, confusion matrix, and checkpoint.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def artifact_entry(path: Path) -> dict:
    return {
        "path": str(path.resolve().relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def cross_split_count(left: pd.DataFrame, right: pd.DataFrame, columns) -> int:
    left_hash = pd.util.hash_pandas_object(left[list(columns)], index=False)
    right_hash = pd.util.hash_pandas_object(right[list(columns)], index=False)
    return int(right_hash.isin(set(left_hash)).sum())


def main():
    args = parse_args()
    data = args.data_root.resolve()
    split_dir = data / "splits"
    serialized_dir = data / "serialized"
    distilbert_run_dir = (
        args.distilbert_run_dir.resolve() if args.distilbert_run_dir else None
    )
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint
        else (
            distilbert_run_dir / "model" / "model.safetensors"
            if distilbert_run_dir
            else data / "models" / "distilbert" / "model.safetensors"
        )
    )
    splits = {
        name: pd.read_parquet(split_dir / f"{name}.parquet")
        for name in ("train", "val", "test")
    }
    missing_features = [c for c in FEATURE_COLUMNS if c not in splits["train"].columns]
    if missing_features:
        raise ValueError(f"missing model features: {missing_features}")

    feature_duplicates = {}
    labeled_duplicates = {}
    serialized_sets = {}
    for feature_set in FEATURE_SET_EXCLUSIONS:
        candidate = serialized_dir if feature_set == "original" else serialized_dir / feature_set
        if all((candidate / f"{name}.parquet").exists() for name in ("train", "val", "test")):
            serialized_sets[feature_set] = {
                name: pd.read_parquet(candidate / f"{name}.parquet")
                for name in ("train", "val", "test")
            }
    serialized_duplicates = {feature_set: {} for feature_set in serialized_sets}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        key = f"{right}_present_in_{left}"
        feature_duplicates[key] = cross_split_count(
            splits[left], splits[right], FEATURE_COLUMNS
        )
        labeled_duplicates[key] = cross_split_count(
            splits[left], splits[right], [*FEATURE_COLUMNS, "Label"]
        )
        for feature_set, serialized in serialized_sets.items():
            serialized_duplicates[feature_set][key] = int(
                serialized[right]["text"].isin(set(serialized[left]["text"])).sum()
            )

    combined = pd.concat(
        [frame.assign(_split=name) for name, frame in splits.items()],
        ignore_index=True,
    )
    feature_hash = pd.util.hash_pandas_object(
        combined[FEATURE_COLUMNS], index=False
    ).rename("_feature_hash")
    grouped = combined.assign(_feature_hash=feature_hash).groupby("_feature_hash")
    group_stats = grouped.agg(
        rows=("_feature_hash", "size"),
        subtype_labels=("Label", "nunique"),
        category_labels=("Category", "nunique"),
        split_count=("_split", "nunique"),
    )

    prediction_check = {"available": False}
    prediction_root = distilbert_run_dir or data
    preds_path = prediction_root / "distilbert_test_preds.npy"
    confusion_path = prediction_root / "confusion_distilbert.csv"
    prediction_records_path = prediction_root / "test_predictions.parquet"
    run_manifest_path = prediction_root / "run_manifest.json"
    if preds_path.exists():
        preds = np.load(preds_path)
        y_true = pd.Categorical(
            splits["test"]["Category"], categories=CLASSES
        ).codes
        if len(preds) != len(y_true):
            prediction_check = {
                "available": True,
                "aligned": False,
                "prediction_rows": len(preds),
                "test_rows": len(y_true),
            }
        else:
            reconstructed = confusion_matrix(
                y_true, preds, labels=np.arange(len(CLASSES))
            )
            prediction_check = {
                "available": True,
                "aligned": True,
                "accuracy": float(np.mean(preds == y_true)),
                "macro_f1": float(f1_score(y_true, preds, average="macro")),
            }
            if confusion_path.exists():
                reported = pd.read_csv(confusion_path, index_col=0).to_numpy()
                prediction_check.update(
                    {
                        "confusion_equal": bool(np.array_equal(reconstructed, reported)),
                        "confusion_abs_difference": int(
                            np.abs(reconstructed - reported).sum()
                        ),
                        "confusion_cells_changed": int(
                            np.sum(reconstructed != reported)
                        ),
                    }
                )
            if prediction_records_path.exists():
                records = pd.read_parquet(prediction_records_path)
                required = {"SampleID", "y_true", "y_pred"}
                missing = required - set(records.columns)
                if missing:
                    raise ValueError(
                        f"{prediction_records_path} is missing {sorted(missing)}"
                    )
                prediction_check.update(
                    {
                        "sample_id_aligned": bool(
                            "SampleID" in splits["test"]
                            and np.array_equal(
                                records["SampleID"].to_numpy(),
                                splits["test"]["SampleID"].to_numpy(),
                            )
                        ),
                        "parquet_prediction_array_equal": bool(
                            np.array_equal(records["y_pred"].to_numpy(), preds)
                        ),
                        "parquet_label_array_equal": bool(
                            np.array_equal(records["y_true"].to_numpy(), y_true)
                        ),
                    }
                )
            if run_manifest_path.exists():
                run_manifest = json.loads(run_manifest_path.read_text())
                prediction_check["run_manifest_consistency_checks"] = run_manifest.get(
                    "consistency_checks", {}
                )

    candidates = [
        *(split_dir / f"{name}.parquet" for name in ("train", "val", "test")),
        *(serialized_dir / f"{name}.parquet" for name in ("train", "val", "test")),
        *(
            serialized_dir / feature_set / f"{name}.parquet"
            for feature_set in FEATURE_SET_EXCLUSIONS
            if feature_set != "original"
            for name in ("train", "val", "test")
        ),
        preds_path,
        confusion_path,
        prediction_root / "transformer_metrics.csv",
        prediction_records_path,
        run_manifest_path,
        checkpoint,
    ]
    artifacts = [artifact_entry(path) for path in candidates if path.exists()]
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "data_root": str(data.relative_to(ROOT)),
        "distilbert_run_dir": (
            str(distilbert_run_dir.relative_to(ROOT)) if distilbert_run_dir else None
        ),
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "provenance_columns": present_provenance(splits["train"].columns),
        "feature_duplicates": feature_duplicates,
        "feature_and_label_duplicates": labeled_duplicates,
        "serialized_duplicates": serialized_duplicates,
        "duplicate_group_summary": {
            "groups": int(len(group_stats)),
            "duplicate_groups": int((group_stats["rows"] > 1).sum()),
            "rows_in_duplicate_groups": int(
                group_stats.loc[group_stats["rows"] > 1, "rows"].sum()
            ),
            "cross_split_groups": int((group_stats["split_count"] > 1).sum()),
            "conflicting_subtype_groups": int(
                (group_stats["subtype_labels"] > 1).sum()
            ),
            "conflicting_category_groups": int(
                (group_stats["category_labels"] > 1).sum()
            ),
            "largest_group_rows": int(group_stats["rows"].max()),
        },
        "prediction_check": prediction_check,
        "artifacts": artifacts,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    duplicate_summary = manifest["duplicate_group_summary"]
    lines = [
        "# Reproducibility audit",
        "",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Data root: `{manifest['data_root']}`",
        f"- Rows: {manifest['split_rows']}",
        f"- Provenance columns present: {manifest['provenance_columns'] or 'none'}",
        "",
        "## Cross-split identity",
        "",
        "| Comparison | Exact features | Exact features + subtype | "
        + " | ".join(f"Serialized `{name}`" for name in serialized_sets)
        + " |",
        "|---|---:|---:|" + "---:|" * len(serialized_sets),
    ]
    for key in feature_duplicates:
        lines.append(
            f"| {key.replace('_', ' ')} | {feature_duplicates[key]} | "
            f"{labeled_duplicates[key]} | "
            + " | ".join(
                str(serialized_duplicates[feature_set][key])
                for feature_set in serialized_sets
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Duplicate and label-conflict groups",
            "",
            f"- Duplicate groups: {duplicate_summary['duplicate_groups']} "
            f"({duplicate_summary['rows_in_duplicate_groups']} rows).",
            f"- Groups crossing splits: {duplicate_summary['cross_split_groups']}.",
            f"- Conflicting subtype groups: {duplicate_summary['conflicting_subtype_groups']}.",
            f"- Conflicting category groups: {duplicate_summary['conflicting_category_groups']}.",
            f"- Largest exact-feature group: {duplicate_summary['largest_group_rows']} rows.",
            "",
            "## DistilBERT prediction consistency",
            "",
        ]
    )
    if not prediction_check["available"]:
        lines.append("No saved prediction array was available.")
    elif not prediction_check["aligned"]:
        lines.append(
            f"Predictions are not aligned: {prediction_check['prediction_rows']} "
            f"predictions for {prediction_check['test_rows']} test rows."
        )
    else:
        lines.extend(
            [
                f"- Accuracy reconstructed from predictions: {prediction_check['accuracy']:.6f}.",
                f"- Macro-F1 reconstructed from predictions: {prediction_check['macro_f1']:.6f}.",
                f"- Confusion matrix identical: {prediction_check.get('confusion_equal', 'n/a')}.",
                f"- Absolute confusion-count difference: "
                f"{prediction_check.get('confusion_abs_difference', 'n/a')}.",
                f"- Confusion cells changed: "
                f"{prediction_check.get('confusion_cells_changed', 'n/a')}.",
                f"- SampleID order identical to test split: "
                f"{prediction_check.get('sample_id_aligned', 'n/a')}.",
                f"- Parquet and NPY predictions identical: "
                f"{prediction_check.get('parquet_prediction_array_equal', 'n/a')}.",
                f"- Parquet and split labels identical: "
                f"{prediction_check.get('parquet_label_array_equal', 'n/a')}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Any non-zero cross-split feature/text identity is a leakage risk. A saved "
            "prediction array that does not reconstruct the reported confusion matrix "
            "must not be mixed with those reported metrics or downstream statistics.",
            "",
            f"Full SHA-256 checksums are stored in `{args.manifest.name}`.",
            "",
        ]
    )
    args.report.write_text("\n".join(lines))
    print(f"wrote {args.manifest}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
