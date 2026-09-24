"""Compute TreeSHAP values for XGBoost on a fixed revised XAI cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from pipeline_config import CLASSES, feature_pairs


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=("original", "no_number"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_shap_shape(values, rows: int, features: int, classes: int):
    if isinstance(values, list):
        array = np.stack(values, axis=-1)
    else:
        array = np.asarray(values)
    if array.shape == (rows, features, classes):
        return array
    if array.shape == (rows, classes, features):
        return np.transpose(array, (0, 2, 1))
    raise ValueError(
        f"unexpected SHAP shape {array.shape}; expected "
        f"({rows}, {features}, {classes}) or ({rows}, {classes}, {features})"
    )


def main():
    args = parse_args()
    model_path = args.model.resolve()
    split_path = args.test_split.resolve()
    cohort_path = args.cohort.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = feature_pairs(args.feature_set)
    columns = [column for column, _ in pairs]
    feature_names = [name for _, name in pairs]
    test = pd.read_parquet(split_path).reset_index(drop=True)
    cohort = pd.read_parquet(cohort_path).sort_values("cohort_order")
    selected = cohort.merge(
        test.reset_index(names="test_row"),
        left_on=["row", "SampleID", "true_category"],
        right_on=["test_row", "SampleID", "Category"],
        validate="one_to_one",
    ).sort_values("cohort_order")
    if len(selected) != len(cohort):
        raise ValueError("cohort did not align completely with the test split")

    x = selected[columns].to_numpy(dtype=np.float32)
    y_true = selected["y_true"].to_numpy()
    model = joblib.load(model_path)
    predictions = model.predict(x)
    if not np.array_equal(predictions, y_true):
        raise ValueError("the supplied cohort contains rows not correctly classified")

    explainer = shap.TreeExplainer(model)
    values = normalize_shap_shape(
        explainer.shap_values(x),
        rows=len(x),
        features=len(columns),
        classes=len(CLASSES),
    )

    per_sample_rows = []
    aggregate_rows = []
    for class_index, class_name in enumerate(CLASSES):
        positions = np.where(y_true == class_index)[0]
        class_values = values[positions, :, class_index]
        l1 = np.abs(class_values).sum(axis=1, keepdims=True)
        normalized = class_values / np.where(l1 == 0, 1.0, l1)
        signed = normalized.mean(axis=0)
        absolute = np.abs(normalized).mean(axis=0)
        order = np.argsort(absolute)[::-1]
        for position, normalized_row in zip(positions, normalized):
            record = {
                "cohort_order": int(selected.iloc[position]["cohort_order"]),
                "class": class_name,
                "row": int(selected.iloc[position]["row"]),
                "SampleID": selected.iloc[position]["SampleID"],
            }
            record.update(
                {
                    feature: float(normalized_row[index])
                    for index, feature in enumerate(feature_names)
                }
            )
            per_sample_rows.append(record)
        for rank, feature_index in enumerate(order, start=1):
            aggregate_rows.append(
                {
                    "class": class_name,
                    "feature": feature_names[feature_index],
                    "signed_mean": float(signed[feature_index]),
                    "abs_mean": float(absolute[feature_index]),
                    "rank": rank,
                }
            )

    per_sample = pd.DataFrame(per_sample_rows).sort_values("cohort_order")
    aggregate = pd.DataFrame(aggregate_rows)
    per_sample_path = output_dir / "shap_per_sample.parquet"
    aggregate_path = output_dir / "shap_feature_attribution.csv"
    per_sample.to_parquet(per_sample_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)

    lines = [
        f"# XGBoost `{args.feature_set}` — TreeSHAP",
        "",
        f"TreeSHAP was evaluated on {len(selected)} fixed cohort rows. Values "
        "explain the true class and are L1-normalized within each sample before "
        "class-level aggregation.",
        "",
        "## Cohort size",
        "",
        per_sample.groupby("class", sort=False).size().rename("rows").reset_index().to_markdown(index=False),
        "",
        "## Top features by class",
        "",
    ]
    for class_name in CLASSES:
        top = aggregate[aggregate["class"] == class_name].nsmallest(8, "rank")
        rendered = ", ".join(
            f"`{row.feature}` ({row.abs_mean:.4f})"
            for row in top.itertuples(index=False)
        )
        lines.append(f"- **{class_name}:** {rendered}")
    lines.append("")
    (output_dir / "shap_summary.md").write_text("\n".join(lines))

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set": args.feature_set,
        "cohort_rows": len(selected),
        "class_counts": {
            str(class_name): int(count)
            for class_name, count in per_sample["class"].value_counts(sort=False).items()
        },
        "feature_count": len(feature_names),
        "inputs": {
            "model": {
                "path": str(model_path.relative_to(ROOT)),
                "sha256": sha256(model_path),
            },
            "test_split": {
                "path": str(split_path.relative_to(ROOT)),
                "sha256": sha256(split_path),
            },
            "cohort": {
                "path": str(cohort_path.relative_to(ROOT)),
                "sha256": sha256(cohort_path),
            },
        },
        "outputs": {
            "per_sample_sha256": sha256(per_sample_path),
            "feature_attribution_sha256": sha256(aggregate_path),
        },
    }
    (output_dir / "shap_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("\n".join(lines))
    print(f"wrote revised TreeSHAP artifacts to {output_dir}")


if __name__ == "__main__":
    main()
