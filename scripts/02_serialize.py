"""Phase 1 — serialize CICIoT2023 flow windows to template text.

Each row becomes a fixed-order "name=value" string over the 39 window-aggregated
features of the regenerated CSV schema (this version exposes protocol-indicator
fractions rather than raw ports — the serialization keeps a 1:1 name mapping to
the original columns so Captum/SHAP attributions can be mapped back).

Floats are trimmed to 4 significant digits to bound token noise; integers stay
integers. Feature order is domain-grouped (protocol context -> TCP flags ->
app-protocol indicators -> size stats -> timing) and identical for every row.

Outputs: data/serialized/{train,val,test}.parquet (text, targets, provenance)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import (
    FEATURE_PAIRS,
    FEATURE_SET_EXCLUSIONS,
    feature_pairs,
    format_value,
    present_provenance,
    serialize_frame,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "data" / "serialized"

# Backwards-compatible export: robustness/XAI scripts import FEATURES.
FEATURES = FEATURE_PAIRS


def fmt(v) -> str:
    """Backwards-compatible alias used by existing robustness scripts."""
    return format_value(v)


def serialize(df: pd.DataFrame, pairs=FEATURES) -> pd.Series:
    return pd.Series(serialize_frame(df, pairs), index=df.index, name="text")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=SPLITS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument(
        "--feature-set", choices=FEATURE_SET_EXCLUSIONS, default="original"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = feature_pairs(args.feature_set)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        df = pd.read_parquet(args.input_dir / f"{split}.parquet")
        out = pd.DataFrame({
            "text": serialize(df, pairs),
            "Label": df["Label"],
            "Category": df["Category"],
        })
        for column in present_provenance(df.columns):
            out[column] = df[column]
        out.to_parquet(args.output_dir / f"{split}.parquet", index=False)
        print(f"{split}: {len(out)} rows serialized")

    metadata = {
        "feature_set": args.feature_set,
        "feature_pairs": pairs,
        "input_dir": str(args.input_dir.resolve()),
    }
    (args.output_dir / "serialization_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    # show one example per selected category + token-length stats
    test = pd.read_parquet(args.output_dir / "test.parquet")
    for cat in ("Benign", "Mirai", "DDoS", "Recon"):
        row = test[test["Category"] == cat].iloc[0]
        print(f"\n[{cat} / {row['Label']}]\n{row['text']}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    sample = test["text"].sample(2000, random_state=42)
    lens = [len(tok(t)["input_ids"]) for t in sample]
    print(f"\nToken lengths (n=2000): mean={np.mean(lens):.0f} "
          f"p95={np.percentile(lens, 95):.0f} max={max(lens)} (model limit 512)")


if __name__ == "__main__":
    main()
