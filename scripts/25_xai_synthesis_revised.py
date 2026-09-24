"""Summarize revised DistilBERT Layer-IG and XGBoost TreeSHAP results."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline_config import CLASSES


FEATURES_OF_INTEREST = ("num", "proto_num", "arp", "icmp", "ssh", "syn")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ig-dir", type=Path, required=True)
    parser.add_argument("--shap-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rank_of(frame: pd.DataFrame, class_name: str, feature: str):
    match = frame[(frame["class"] == class_name) & (frame["feature"] == feature)]
    return int(match.iloc[0]["rank"]) if len(match) else None


def main():
    args = parse_args()
    ig_dir = args.ig_dir.resolve()
    shap_dir = args.shap_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ig_manifest = json.loads((ig_dir / "captum_manifest.json").read_text())
    shap_manifest = json.loads((shap_dir / "shap_manifest.json").read_text())
    if ig_manifest["inputs"]["cohort"]["sha256"] != shap_manifest["inputs"]["cohort"]["sha256"]:
        raise ValueError("Layer-IG and TreeSHAP did not use the same cohort")
    if ig_manifest["cohort_rows"] != shap_manifest["cohort_rows"]:
        raise ValueError("Layer-IG and TreeSHAP cohort sizes differ")

    ig = pd.read_csv(ig_dir / "captum_feature_attribution.csv")
    shap = pd.read_csv(shap_dir / "shap_feature_attribution.csv")
    agreement_rows = []
    rank_rows = []
    for class_name in CLASSES:
        ig_class = ig[ig["class"] == class_name].set_index("feature")
        shap_class = shap[shap["class"] == class_name].set_index("feature")
        shared = ig_class.index.intersection(shap_class.index)
        if len(shared) < 2:
            raise ValueError(f"insufficient shared features for {class_name}")
        rho = spearmanr(
            ig_class.loc[shared, "abs_mean"],
            shap_class.loc[shared, "abs_mean"],
        ).correlation
        ig_top5 = set(ig_class.nsmallest(5, "rank").index)
        shap_top5 = set(shap_class.nsmallest(5, "rank").index)
        agreement_rows.append(
            {
                "class": class_name,
                "shared_features": len(shared),
                "spearman_abs_importance": float(rho),
                "top5_overlap_count": len(ig_top5 & shap_top5),
                "top5_overlap_fraction": len(ig_top5 & shap_top5) / 5,
                "ig_top1": ig_class.nsmallest(1, "rank").index[0],
                "shap_top1": shap_class.nsmallest(1, "rank").index[0],
            }
        )
        for feature in FEATURES_OF_INTEREST:
            rank_rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "ig_rank": rank_of(ig, class_name, feature),
                    "shap_rank": rank_of(shap, class_name, feature),
                }
            )

    agreement = pd.DataFrame(agreement_rows)
    ranks = pd.DataFrame(rank_rows)
    agreement_path = output_dir / "xai_agreement.csv"
    ranks_path = output_dir / "xai_feature_ranks.csv"
    agreement.to_csv(agreement_path, index=False)
    ranks.to_csv(ranks_path, index=False)

    lines = [
        "# Revised cross-model XAI comparison",
        "",
        f"Layer Integrated Gradients and TreeSHAP use the same "
        f"{ig_manifest['cohort_rows']} `SampleID`-aligned rows. Attributions explain "
        "the true class and are L1-normalized per sample before aggregation.",
        "",
        "| Class | Spearman | Top-5 overlap | IG top feature | SHAP top feature |",
        "|---|---:|---:|---|---|",
    ]
    for _, row in agreement.iterrows():
        lines.append(
            f"| {row['class']} | {row['spearman_abs_importance']:+.3f} | "
            f"{int(row['top5_overlap_count'])}/5 | `{row['ig_top1']}` | "
            f"`{row['shap_top1']}` |"
        )
    lines.extend(
        [
            "",
            f"Mean Spearman correlation: {agreement['spearman_abs_importance'].mean():+.3f}. "
            f"Mean top-5 overlap: {agreement['top5_overlap_fraction'].mean():.1%}.",
            "",
            "Agreement between rankings is descriptive rather than a test that one "
            "method validates the other. Differences can reflect the models' decision "
            "rules as well as the distinct attribution methods. Domain interpretation "
            "should be added only after checking the relevant verified fact sheets.",
            "",
        ]
    )
    summary_path = output_dir / "xai_comparison.md"
    summary_path.write_text("\n".join(lines))
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_rows": ig_manifest["cohort_rows"],
        "cohort_sha256": ig_manifest["inputs"]["cohort"]["sha256"],
        "inputs": {
            "ig_manifest_sha256": sha256(ig_dir / "captum_manifest.json"),
            "ig_attribution_sha256": sha256(
                ig_dir / "captum_feature_attribution.csv"
            ),
            "shap_manifest_sha256": sha256(shap_dir / "shap_manifest.json"),
            "shap_attribution_sha256": sha256(
                shap_dir / "shap_feature_attribution.csv"
            ),
        },
        "outputs": {
            "agreement_sha256": sha256(agreement_path),
            "ranks_sha256": sha256(ranks_path),
            "summary_sha256": sha256(summary_path),
        },
    }
    (output_dir / "xai_synthesis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("\n".join(lines))
    print(f"wrote revised XAI synthesis to {output_dir}")


if __name__ == "__main__":
    main()
