"""Bootstrap stability of the per-class attribution rankings.

Rankings in Section 4.2 are means over a finite cohort, so a feature can top a
class by sampling luck. This resamples the cohort rows within each class, with
replacement, and records how often the observed top-1 feature stays top-1 and
how often it stays inside the top-3. No model inference is involved: it reads
the saved per-sample attributions, so it costs seconds and cannot perturb any
existing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import CLASSES, feature_pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True, help="method name used in outputs")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.resamples <= 0:
        raise ValueError("resample count must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample = pd.read_parquet(args.per_sample)
    observed = pd.read_csv(args.attribution)
    names = [name for _, name in feature_pairs("original")]
    features = [name for name in names if name in per_sample.columns]
    if len(features) < 2:
        raise ValueError("per-sample file does not carry the serialized field columns")

    rng = np.random.default_rng(args.seed)
    rows = []
    for class_name in CLASSES:
        subset = per_sample[per_sample["class"] == class_name]
        if subset.empty:
            raise ValueError(f"no attribution rows for {class_name}")
        values = np.abs(subset[features].to_numpy(np.float64))
        top1 = observed[observed["class"] == class_name].nsmallest(1, "rank")
        observed_top1 = top1["feature"].iloc[0]
        if observed_top1 not in features:
            raise ValueError(f"observed top feature {observed_top1} missing from per-sample data")
        target = features.index(observed_top1)

        draws = rng.integers(0, len(values), size=(args.resamples, len(values)))
        means = values[draws].mean(axis=1)
        order = np.argsort(-means, axis=1)
        stays_top1 = float(np.mean(order[:, 0] == target))
        stays_top3 = float(np.mean(np.any(order[:, :3] == target, axis=1)))
        rows.append(
            {
                "class": class_name,
                "n": len(values),
                "observed_top1": observed_top1,
                "p_stays_top1": stays_top1,
                "p_stays_top3": stays_top3,
            }
        )

    result = pd.DataFrame(rows)
    csv_path = output_dir / f"{args.label}_rank_stability.csv"
    result.to_csv(csv_path, index=False)

    lines = [
        f"# Bootstrap stability of the {args.label} per-class rankings",
        "",
        f"Each class's cohort rows were resampled with replacement {args.resamples:,} "
        "times and the mean absolute attribution recomputed. The table reports how "
        "often the observed top feature remains first, and how often it remains "
        "within the first three.",
        "",
        "| Class | n | Observed top-1 | Stays top-1 | Stays top-3 |",
        "|---|---:|---|---:|---:|",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row._0 if not hasattr(row, 'class') else getattr(row, 'class')} "
            f"| {row.n} | `{row.observed_top1}` | {row.p_stays_top1:.2f} | "
            f"{row.p_stays_top3:.2f} |"
        )
    lines.extend(
        [
            "",
            "A low value means the class's leading feature is decided by which rows "
            "entered the cohort and should not be reported as that class's signature. "
            "Stability of the ranking says nothing about whether the feature is "
            "causally used; it bounds only the sampling error of the estimate.",
            "",
        ]
    )
    summary_path = output_dir / f"{args.label}_rank_stability.md"
    summary_path.write_text("\n".join(lines))

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "resamples": args.resamples,
        "seed": args.seed,
        "feature_count": len(features),
        "inputs": {
            "per_sample_sha256": sha256(args.per_sample.resolve()),
            "attribution_sha256": sha256(args.attribution.resolve()),
        },
        "outputs": {
            "csv_sha256": sha256(csv_path),
            "summary_sha256": sha256(summary_path),
        },
    }
    (output_dir / f"{args.label}_rank_stability_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
