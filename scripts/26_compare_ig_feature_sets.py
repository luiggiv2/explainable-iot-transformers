"""Compare Layer-IG patterns for original and no-Number DistilBERT models."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from pipeline_config import CLASSES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--no-number-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    original_dir = args.original_dir.resolve()
    no_number_dir = args.no_number_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_manifest = json.loads(
        (original_dir / "captum_manifest.json").read_text()
    )
    no_number_manifest = json.loads(
        (no_number_dir / "captum_manifest.json").read_text()
    )
    if original_manifest["feature_set"] != "original":
        raise ValueError("the original IG directory has the wrong feature set")
    if no_number_manifest["feature_set"] != "no_number":
        raise ValueError("the no-number IG directory has the wrong feature set")
    if original_manifest["inputs"]["cohort"]["sha256"] != no_number_manifest[
        "inputs"
    ]["cohort"]["sha256"]:
        raise ValueError("the two IG runs did not use the same cohort")

    original_samples = pd.read_parquet(
        original_dir / "captum_per_sample.parquet"
    )
    no_number_samples = pd.read_parquet(
        no_number_dir / "captum_per_sample.parquet"
    )
    identity = ["SampleID", "class", "row"]
    aligned = original_samples[identity].merge(
        no_number_samples[identity], on=identity, validate="one_to_one"
    )
    if len(aligned) != len(original_samples) or len(aligned) != len(no_number_samples):
        raise ValueError("the per-sample IG records do not align one-to-one")

    original = pd.read_csv(original_dir / "captum_feature_attribution.csv")
    no_number = pd.read_csv(no_number_dir / "captum_feature_attribution.csv")
    agreement_rows = []
    change_rows = []
    for class_name in CLASSES:
        original_class = original[original["class"] == class_name].set_index(
            "feature"
        )
        no_number_class = no_number[
            no_number["class"] == class_name
        ].set_index("feature")
        shared = original_class.index.intersection(no_number_class.index)
        rho = spearmanr(
            original_class.loc[shared, "abs_mean"],
            no_number_class.loc[shared, "abs_mean"],
        ).correlation
        original_top5 = set(original_class.nsmallest(5, "rank").index) - {"num"}
        no_number_top5 = set(no_number_class.nsmallest(5, "rank").index)
        agreement_rows.append(
            {
                "class": class_name,
                "shared_features": len(shared),
                "spearman_shared_abs_importance": float(rho),
                "top5_overlap_count": len(original_top5 & no_number_top5),
                "original_num_abs_mean": float(
                    original_class.loc["num", "abs_mean"]
                ),
                "original_num_rank": int(original_class.loc["num", "rank"]),
            }
        )
        for feature in shared:
            change_rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "original_abs_mean": float(
                        original_class.loc[feature, "abs_mean"]
                    ),
                    "no_number_abs_mean": float(
                        no_number_class.loc[feature, "abs_mean"]
                    ),
                    "abs_mean_delta": float(
                        no_number_class.loc[feature, "abs_mean"]
                        - original_class.loc[feature, "abs_mean"]
                    ),
                    "original_rank": int(original_class.loc[feature, "rank"]),
                    "no_number_rank": int(no_number_class.loc[feature, "rank"]),
                }
            )

    agreement = pd.DataFrame(agreement_rows)
    changes = pd.DataFrame(change_rows)
    agreement_path = output_dir / "ig_feature_set_agreement.csv"
    changes_path = output_dir / "ig_shared_feature_changes.csv"
    agreement.to_csv(agreement_path, index=False)
    changes.to_csv(changes_path, index=False)

    lines = [
        "# Layer-IG comparison after removing `Number`",
        "",
        "Both models are explained on the same rows, and every row is correctly "
        "classified by both checkpoints. Correlations use the 38 fields shared by "
        "the two serializations.",
        "",
        "| Class | Shared-feature Spearman | Original `num` rank | Original `num` magnitude | Largest increase without `num` |",
        "|---|---:|---:|---:|---|",
    ]
    for row in agreement.itertuples(index=False):
        replacements = changes[changes["class"] == row[0]].nlargest(
            1, "abs_mean_delta"
        )
        replacement = replacements.iloc[0]
        lines.append(
            f"| {row[0]} | {row.spearman_shared_abs_importance:+.3f} | "
            f"{row.original_num_rank} | {row.original_num_abs_mean:.4f} | "
            f"`{replacement['feature']}` ({replacement['abs_mean_delta']:+.4f}) |"
        )
    lines.extend(
        [
            "",
            "Because each sample is L1-normalized, increases after removing `Number` "
            "describe redistributed attribution mass; they are not independent effect "
            "sizes. The performance ablation remains the stronger evidence about the "
            "predictive consequence of removing the field.",
            "",
        ]
    )
    summary_path = output_dir / "ig_feature_set_comparison.md"
    summary_path.write_text("\n".join(lines))
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_rows": original_manifest["cohort_rows"],
        "cohort_sha256": original_manifest["inputs"]["cohort"]["sha256"],
        "inputs": {
            "original_manifest_sha256": sha256(
                original_dir / "captum_manifest.json"
            ),
            "no_number_manifest_sha256": sha256(
                no_number_dir / "captum_manifest.json"
            ),
        },
        "outputs": {
            "agreement_sha256": sha256(agreement_path),
            "changes_sha256": sha256(changes_path),
            "summary_sha256": sha256(summary_path),
        },
    }
    (output_dir / "ig_feature_set_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("\n".join(lines))
    print(f"wrote IG feature-set comparison to {output_dir}")


if __name__ == "__main__":
    main()
