"""Phase 1 — stratified subset of CICIoT2023 (regenerated 39-feature CSV version).

Sampling design (approved 2026-07-16): ~60k rows, category allocation
proportional to sqrt(category size) with a 1,500-row floor for minority
categories (Web-based, BruteForce); within each category, subtypes are
sampled proportionally to their true frequency. Fixed seed throughout.

Two-pass exact sampling: pass 1 (already done, data/class_distribution_raw.csv)
gives per-subtype counts; here we draw per-subtype ordinal indices and stream
the 63 Merged CSVs in filename order, keeping rows whose within-subtype
ordinal was drawn. Malformed rows (label not in the 34 known labels) are dropped.

Outputs (all under data/):
  subset_60k.parquet            39 features + targets + provenance columns
  splits/{train,val,test}.parquet   grouped-stratified, approximately 70/15/15
  subset_composition.md         per-category/subtype × split table
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from pipeline_config import SEED, feature_pairs, serialize_frame

TOTAL_TARGET = 60_000
FLOOR = 1_500

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"

CATEGORY_MAP = {
    # DDoS (12)
    "DDOS-ICMP_FLOOD": "DDoS", "DDOS-UDP_FLOOD": "DDoS", "DDOS-TCP_FLOOD": "DDoS",
    "DDOS-PSHACK_FLOOD": "DDoS", "DDOS-SYN_FLOOD": "DDoS", "DDOS-RSTFINFLOOD": "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS", "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-UDP_FRAGMENTATION": "DDoS", "DDOS-ACK_FRAGMENTATION": "DDoS",
    "DDOS-HTTP_FLOOD": "DDoS", "DDOS-SLOWLORIS": "DDoS",
    # DoS (4)
    "DOS-UDP_FLOOD": "DoS", "DOS-TCP_FLOOD": "DoS", "DOS-SYN_FLOOD": "DoS",
    "DOS-HTTP_FLOOD": "DoS",
    # Mirai (3)
    "MIRAI-GREETH_FLOOD": "Mirai", "MIRAI-UDPPLAIN": "Mirai", "MIRAI-GREIP_FLOOD": "Mirai",
    # Benign
    "BENIGN": "Benign",
    # Spoofing (2)
    "MITM-ARPSPOOFING": "Spoofing", "DNS_SPOOFING": "Spoofing",
    # Recon (5) — VulnerabilityScan belongs to Recon in the official grouping
    "VULNERABILITYSCAN": "Recon", "RECON-HOSTDISCOVERY": "Recon", "RECON-OSSCAN": "Recon",
    "RECON-PORTSCAN": "Recon", "RECON-PINGSWEEP": "Recon",
    # Web-based (6)
    "BROWSERHIJACKING": "Web", "COMMANDINJECTION": "Web", "SQLINJECTION": "Web",
    "XSS": "Web", "BACKDOOR_MALWARE": "Web", "UPLOADING_ATTACK": "Web",
    # Brute force
    "DICTIONARYBRUTEFORCE": "BruteForce",
}


def category_targets(cat_counts: dict[str, int]) -> dict[str, int]:
    w = {c: np.sqrt(n) for c, n in cat_counts.items()}
    raw = {c: TOTAL_TARGET * w[c] / sum(w.values()) for c in w}
    floored = {c for c, v in raw.items() if v < FLOOR}
    targets = {c: FLOOR for c in floored}
    rest = TOTAL_TARGET - FLOOR * len(floored)
    w_rest = {c: w[c] for c in w if c not in floored}
    for c in w_rest:
        targets[c] = int(round(rest * w_rest[c] / sum(w_rest.values())))
    return targets


def subtype_targets(cat_targets, subtype_counts):
    """Allocate each category's target across its subtypes proportionally."""
    targets = {}
    for cat, cat_target in cat_targets.items():
        subs = {s: n for s, n in subtype_counts.items() if CATEGORY_MAP[s] == cat}
        total = sum(subs.values())
        alloc = {s: max(1, int(round(cat_target * n / total))) for s, n in subs.items()}
        # fix rounding drift on the largest subtype, capped by availability
        drift = cat_target - sum(alloc.values())
        biggest = max(subs, key=subs.get)
        alloc[biggest] = min(subs[biggest], alloc[biggest] + drift)
        targets.update(alloc)
    return targets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="Root for subset_60k.parquet, splits/, and audit artifacts.",
    )
    return parser.parse_args()


def assign_grouped_splits(subset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Keep feature-identical rows together while preserving subtype balance.

    A seven-fold grouped split first reserves one fold for test. A second
    six-fold grouped split reserves one fold from the remainder for validation,
    yielding approximately 5/7, 1/7, 1/7 (71.4/14.3/14.3). The smallest sampled
    subtype has too few rows for a leakage-safe 20-fold 70/15/15 construction.
    FeatureGroup is computed before splitting, so exact duplicate windows can
    never leak across partitions even when their labels conflict.
    """
    # Group on the exact text visible to the most strongly ablated model. This
    # is stricter than grouping exact float rows: four-significant-digit
    # serialization can otherwise collapse nearby numeric rows into the same
    # input string across splits. Equality under a richer feature set implies
    # equality after removing Number and Protocol Type, so one grouping protects
    # all four registered ablation configurations.
    grouping_text = serialize_frame(
        subset, feature_pairs("no_number_protocol")
    )
    subset = subset.copy()
    subset["FeatureGroup"] = [
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in grouping_text
    ]

    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    placeholder = np.zeros((len(subset), 1), dtype=np.int8)
    remaining_idx, test_idx = next(
        outer.split(placeholder, subset["Label"], groups=subset["FeatureGroup"])
    )
    remaining = subset.iloc[remaining_idx]
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED + 1)
    inner_placeholder = np.zeros((len(remaining), 1), dtype=np.int8)
    train_pos, val_pos = next(
        inner.split(
            inner_placeholder,
            remaining["Label"],
            groups=remaining["FeatureGroup"],
        )
    )
    train_idx = remaining_idx[train_pos]
    val_idx = remaining_idx[val_pos]
    splits = {
        "train": subset.iloc[train_idx].reset_index(drop=True),
        "val": subset.iloc[val_idx].reset_index(drop=True),
        "test": subset.iloc[test_idx].reset_index(drop=True),
    }

    owners = {}
    for name, frame in splits.items():
        for group in frame["FeatureGroup"].unique():
            assert group not in owners, f"feature group {group} leaks across splits"
            owners[group] = name
    return splits


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    dist = pd.read_csv(OUT / "class_distribution_raw.csv", header=None,
                       names=["label", "count"])
    subtype_counts = {r.label: int(r.count) for r in dist.itertuples()
                      if r.label in CATEGORY_MAP}
    dropped = int(dist.loc[~dist.label.isin([*CATEGORY_MAP, "TOTAL"]), "count"].sum())
    print(f"34 known labels; {dropped} malformed rows will be skipped")

    cat_counts = {}
    for s, n in subtype_counts.items():
        cat_counts[CATEGORY_MAP[s]] = cat_counts.get(CATEGORY_MAP[s], 0) + n
    cat_t = category_targets(cat_counts)
    sub_t = subtype_targets(cat_t, subtype_counts)
    print("Category targets:", json.dumps(cat_t, indent=0))
    assert abs(sum(sub_t.values()) - TOTAL_TARGET) <= len(cat_t)

    # draw within-subtype ordinals to keep
    keep = {s: set(rng.choice(subtype_counts[s], size=t, replace=False).tolist())
            for s, t in sub_t.items()}

    # stream files in fixed order, select rows by per-subtype running ordinal
    seen = dict.fromkeys(subtype_counts, 0)
    picked = []
    files = sorted(RAW.glob("Merged*.csv"))
    assert len(files) == 63, f"expected 63 files, found {len(files)}"
    for f in files:
        source_row = 0
        for chunk in pd.read_csv(f, chunksize=500_000, dtype={"Label": str}):
            chunk["SourceFile"] = f.name
            chunk["SourceRow"] = np.arange(source_row, source_row + len(chunk), dtype=np.int64)
            source_row += len(chunk)
            chunk = chunk[chunk["Label"].isin(CATEGORY_MAP)]
            for label, grp in chunk.groupby("Label", sort=False):
                base = seen[label]
                ords = np.arange(base, base + len(grp))
                mask = np.isin(ords, list(keep[label]))
                if mask.any():
                    picked.append(grp.iloc[mask])
                seen[label] += len(grp)
        print(f"{f.name} done, picked so far: {sum(len(p) for p in picked)}")

    subset = pd.concat(picked, ignore_index=True)
    num = subset.select_dtypes("number")
    nonfinite = (np.isinf(num).any(axis=1) | num.isna().any(axis=1))
    n_nonfinite = int(nonfinite.sum())
    if n_nonfinite:
        print(f"dropping {n_nonfinite} rows with inf/NaN feature values")
        subset = subset[~nonfinite].reset_index(drop=True)
    subset["Category"] = subset["Label"].map(CATEGORY_MAP)
    subset["SampleID"] = (
        subset["SourceFile"].str.removesuffix(".csv")
        + ":"
        + subset["SourceRow"].astype(str)
    )
    assert subset["SampleID"].is_unique
    # shuffle so downstream tools never see file order
    subset = subset.sample(frac=1, random_state=SEED).reset_index(drop=True)
    splits = assign_grouped_splits(subset)
    # assign_grouped_splits adds FeatureGroup; preserve it in the combined file
    subset = pd.concat(splits.values(), ignore_index=True)
    subset.to_parquet(out / "subset_60k.parquet", index=False)
    print(f"subset: {len(subset)} rows -> {out / 'subset_60k.parquet'}")

    # grouped-stratified 70/15/15 by subtype label
    (out / "splits").mkdir(exist_ok=True)
    for name, df in splits.items():
        df.to_parquet(out / "splits" / f"{name}.parquet", index=False)
        print(f"{name}: {len(df)} rows")

    # composition table for traceability
    rows = []
    for s in sorted(subtype_counts, key=lambda s: (CATEGORY_MAP[s], s)):
        rows.append({
            "Category": CATEGORY_MAP[s], "Subtype": s,
            "Full dataset": subtype_counts[s],
            **{n: int((df["Label"] == s).sum()) for n, df in splits.items()},
        })
    comp = pd.DataFrame(rows)
    cat_rows = comp.groupby("Category", as_index=False).sum(numeric_only=True)
    cat_rows.insert(1, "Subtype", "(all)")
    lines = [
        "# Subset composition — CICIoT2023 stratified sample",
        "",
        f"Seed {SEED}; sqrt-proportional category allocation, floor {FLOOR}; "
        "subtypes proportional within category; nested grouped-stratified split "
        "(approximately 5/7, 1/7, 1/7 for train/validation/test). Exact feature "
        "duplicates and serialization-equivalent rows are confined to one split. "
        "Feature groups use the no-Number/no-Protocol serialization, which also "
        "protects all richer registered feature sets.",
        f"Malformed rows dropped from raw scan: {dropped} (of 45,019,243). "
        f"Rows dropped for non-finite feature values (inf/NaN): {n_nonfinite}.",
        "",
        "## Per category",
        cat_rows.to_markdown(index=False),
        "",
        "## Per subtype",
        comp.to_markdown(index=False),
        "",
    ]
    (out / "subset_composition.md").write_text("\n".join(lines))
    print(f"wrote {out / 'subset_composition.md'}")


if __name__ == "__main__":
    main()
