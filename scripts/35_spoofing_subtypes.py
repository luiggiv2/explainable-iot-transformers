"""ARP-field distribution and attribution ranks for the two Spoofing subtypes.

The Spoofing category of CICIoT2023 merges two subtypes, MITM-ARPSPOOFING and
DNS_SPOOFING. A statement about how often a Spoofing window carries ARP traffic
is therefore only meaningful per subtype. This script reports, for the train
and test splits and for the fixed 880-row XAI cohort, the distribution of the
`ARP` indicator (fraction of windows with ARP == 0, median, 75th percentile,
maximum) and the mean protocol-indicator composition, separately for each
subtype. It then re-aggregates the saved per-sample attributions (Layer IG on
DistilBERT, TreeSHAP on XGBoost) by subtype and reports the rank of `arp` and
`num` and the top-5 fields for each.

No model inference is involved: only split files and saved per-sample
attribution tables are read, so the script is torch-free and costs seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import feature_pairs


ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "data" / "revision"
SUBTYPES = ["MITM-ARPSPOOFING", "DNS_SPOOFING"]
INDICATORS = ["ARP", "TCP", "UDP", "DNS", "HTTP", "HTTPS", "ICMP", "IPv", "LLC"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=REV / "splits")
    parser.add_argument("--cohort", type=Path, default=REV / "xai" / "cohorts" / "original_xgb.parquet")
    parser.add_argument(
        "--captum", type=Path,
        default=REV / "xai" / "main" / "distilbert" / "captum_per_sample.parquet",
    )
    parser.add_argument(
        "--shap", type=Path,
        default=REV / "xai" / "main" / "xgboost" / "shap_per_sample.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REV / "results" / "spoofing_subtypes"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arp_rows(frame: pd.DataFrame, scope: str) -> list[dict]:
    rows = []
    spoof = frame[frame["Category"] == "Spoofing"]
    groups = [("Spoofing (all)", spoof)] + [
        (subtype, spoof[spoof["Label"] == subtype]) for subtype in SUBTYPES
    ]
    if (frame["Category"] != "Spoofing").any():
        groups.append(("All other categories (reference)", frame[frame["Category"] != "Spoofing"]))
        groups.append(("Benign (reference)", frame[frame["Category"] == "Benign"]))
    for name, subset in groups:
        arp = subset["ARP"].astype(float)
        row = {
            "scope": scope,
            "group": name,
            "n": int(len(subset)),
            "frac_arp_zero": float((arp == 0).mean()),
            "frac_arp_positive": float((arp > 0).mean()),
            "arp_median": float(arp.median()),
            "arp_p75": float(arp.quantile(0.75)),
            "arp_max": float(arp.max()),
            "arp_mean": float(arp.mean()),
        }
        for column in INDICATORS:
            row[f"mean_{column}"] = float(subset[column].astype(float).mean())
        rows.append(row)
    return rows


def subtype_ranks(per_sample: pd.DataFrame, labels: pd.DataFrame, method: str, names) -> list[dict]:
    merged = per_sample.merge(labels, on="SampleID", validate="one_to_one")
    merged = merged[merged["class"] == "Spoofing"]
    rows = []
    groups = [("Spoofing (all)", merged)] + [
        (subtype, merged[merged["Label"] == subtype]) for subtype in SUBTYPES
    ]
    for name, subset in groups:
        if subset.empty:
            continue
        magnitude = subset[names].abs().mean(axis=0)
        order = magnitude.sort_values(ascending=False)
        ranks = {feature: position for position, feature in enumerate(order.index, start=1)}
        rows.append(
            {
                "method": method,
                "group": name,
                "n": int(len(subset)),
                "rank_arp": ranks["arp"],
                "abs_arp": float(magnitude["arp"]),
                "rank_num": ranks["num"],
                "abs_num": float(magnitude["num"]),
                "top5": ", ".join(order.index[:5]),
            }
        )
    return rows


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [name for _, name in feature_pairs("original")]

    inputs = {}
    distribution = []
    for split in ("train", "test"):
        path = args.split_dir / f"{split}.parquet"
        inputs[f"{split}_split"] = path
        frame = pd.read_parquet(path)
        distribution += arp_rows(frame, split)
        if split == "test":
            test = frame

    cohort = pd.read_parquet(args.cohort)
    inputs["cohort"] = args.cohort
    cohort_rows = cohort.merge(
        test[["SampleID", "Label", "Category"] + INDICATORS],
        on="SampleID",
        validate="one_to_one",
    )
    distribution += arp_rows(cohort_rows, "xai_cohort")
    dist = pd.DataFrame(distribution)

    labels = test[["SampleID", "Label"]]
    ranks = []
    for method, path in (("Layer-IG (DistilBERT)", args.captum), ("TreeSHAP (XGBoost)", args.shap)):
        inputs[method] = path
        ranks += subtype_ranks(pd.read_parquet(path), labels, method, names)
    rank_frame = pd.DataFrame(ranks)

    dist_path = output_dir / "spoofing_arp_by_subtype.csv"
    rank_path = output_dir / "spoofing_attribution_by_subtype.csv"
    dist.to_csv(dist_path, index=False)
    rank_frame.to_csv(rank_path, index=False)

    lines = [
        "# Spoofing by subtype: ARP presence and attribution ranks",
        "",
        "The CICIoT2023 Spoofing category merges MITM-ARPSPOOFING and DNS_SPOOFING. "
        "`ARP` is the per-window fraction of packets carrying ARP.",
        "",
        "## ARP-field distribution",
        "",
        "| Scope | Group | n | ARP = 0 | ARP > 0 | median | p75 | max | mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dist.itertuples(index=False):
        lines.append(
            f"| {row.scope} | {row.group} | {row.n} | {row.frac_arp_zero:.1%} | "
            f"{row.frac_arp_positive:.1%} | {row.arp_median:.2f} | {row.arp_p75:.2f} | "
            f"{row.arp_max:.2f} | {row.arp_mean:.3f} |"
        )
    lines += [
        "",
        "## Mean protocol-indicator composition (per-window fractions)",
        "",
        "| Scope | Group | " + " | ".join(INDICATORS) + " |",
        "|---|---|" + "---:|" * len(INDICATORS),
    ]
    for row in dist.to_dict("records"):
        lines.append(
            f"| {row['scope']} | {row['group']} | "
            + " | ".join(f"{row['mean_' + c]:.3f}" for c in INDICATORS)
            + " |"
        )
    lines += [
        "",
        "## Attribution rank of `arp` and `num` by subtype (XAI cohort, 39 fields)",
        "",
        "Ranks are by mean |per-sample L1-normalized attribution| within the group.",
        "",
        "| Method | Group | n | rank `arp` | mean abs `arp` | rank `num` | mean abs `num` | top-5 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rank_frame.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.group} | {row.n} | {row.rank_arp} | {row.abs_arp:.4f} | "
            f"{row.rank_num} | {row.abs_num:.4f} | {row.top5} |"
        )
    lines.append("")
    summary_path = output_dir / "spoofing_subtypes.md"
    summary_path.write_text("\n".join(lines))

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/35_spoofing_subtypes.py",
        "inputs": {key: {"path": str(Path(p).resolve().relative_to(ROOT)), "sha256": sha256(Path(p))}
                   for key, p in inputs.items()},
        "outputs": {
            path.name: sha256(path) for path in (dist_path, rank_path, summary_path)
        },
    }
    (output_dir / "spoofing_subtypes_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
