"""Gradient-free field-masking cross-check on a fixed revised XAI cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline_config import CLASSES, feature_pairs


ROOT = Path(__file__).resolve().parents[1]
FIELD_RE = re.compile(r"([A-Za-z_]+)=")
MAX_LEN = 224


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=("original", "no_number"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--serialized-test", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--ig-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested, but MPS is unavailable")
    return torch.device(requested)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_field_map(text, offsets, names):
    matches = [(match.group(1), match.start()) for match in FIELD_RE.finditer(text)]
    found_names = [name for name, _ in matches]
    if found_names != names:
        raise ValueError("serialized field names or order do not match the feature set")
    spans = [
        (start, matches[index + 1][1] if index + 1 < len(matches) else len(text))
        for index, (_, start) in enumerate(matches)
    ]
    mapping = np.full(len(offsets), -1, dtype=np.int64)
    for token_index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        for feature_index, (feature_start, feature_end) in enumerate(spans):
            if feature_start <= start < feature_end:
                mapping[token_index] = feature_index
                break
    return mapping


def main():
    args = parse_args()
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    model_dir = args.model_dir.resolve()
    serialized_path = args.serialized_test.resolve()
    cohort_path = args.cohort.resolve()
    ig_dir = args.ig_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "ablation_per_sample_progress.parquet"
    final_path = output_dir / "ablation_per_sample.parquet"
    if final_path.exists():
        raise FileExistsError(f"completed output already exists at {final_path}")
    if progress_path.exists() and not args.resume:
        raise FileExistsError(
            f"progress exists at {progress_path}; pass --resume to continue"
        )

    ig_manifest = json.loads((ig_dir / "captum_manifest.json").read_text())
    if ig_manifest["inputs"]["cohort"]["sha256"] != sha256(cohort_path):
        raise ValueError("the masking cross-check and IG cohort differ")

    names = [name for _, name in feature_pairs(args.feature_set)]
    serialized = pd.read_parquet(serialized_path).reset_index(drop=True)
    cohort = pd.read_parquet(cohort_path).sort_values("cohort_order")
    selected = cohort.merge(
        serialized.reset_index(names="serialized_row"),
        left_on=["row", "SampleID", "true_category"],
        right_on=["serialized_row", "SampleID", "Category"],
        validate="one_to_one",
    ).sort_values("cohort_order")
    if len(selected) != len(cohort):
        raise ValueError("cohort did not align completely with serialized test data")
    if args.dry_run:
        selected = selected.head(1)

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    pad_id = tokenizer.pad_token_id

    @torch.no_grad()
    def logits(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    records = (
        pd.read_parquet(progress_path).to_dict("records")
        if progress_path.exists()
        else []
    )
    completed = {record["SampleID"] for record in records}
    pending = selected[~selected["SampleID"].isin(completed)]
    print(
        f"device {device}; feature_set {args.feature_set}; cohort {len(selected)}; "
        f"completed {len(completed)}; pending {len(pending)}",
        flush=True,
    )

    for processed, row in enumerate(pending.itertuples(index=False), start=1):
        encoded = tokenizer(
            row.text,
            truncation=True,
            max_length=MAX_LEN,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        mapping = token_field_map(row.text, offsets, names)
        base = logits(
            input_ids.to(device), attention_mask.to(device)
        )[0, int(row.y_true)].item()
        masked = input_ids.repeat(len(names), 1).clone()
        for feature_index in range(len(names)):
            masked[feature_index, torch.from_numpy(mapping == feature_index)] = pad_id
        masked_attention = attention_mask.repeat(len(names), 1)
        masked_logits = logits(
            masked.to(device), masked_attention.to(device)
        )[:, int(row.y_true)].cpu().numpy()
        drops = base - masked_logits
        total = np.abs(drops).sum() or 1.0
        record = {
            "class": row.true_category,
            "row": int(row.row),
            "SampleID": row.SampleID,
        }
        record.update(
            {
                feature: float(drops[index] / total)
                for index, feature in enumerate(names)
            }
        )
        records.append(record)
        if processed % args.save_every == 0:
            pd.DataFrame(records).fillna(0.0).to_parquet(progress_path, index=False)
            print(f"  masked {len(records)}/{len(selected)} rows", flush=True)

    if args.dry_run:
        print(pd.DataFrame(records).to_string(index=False))
        return

    per_sample = pd.DataFrame(records).fillna(0.0)
    if len(per_sample) != len(selected) or not per_sample["SampleID"].is_unique:
        raise ValueError("final masking rows are incomplete or duplicated")
    per_sample = selected[["cohort_order", "SampleID"]].merge(
        per_sample, on="SampleID", validate="one_to_one"
    ).sort_values("cohort_order")
    per_sample.to_parquet(progress_path, index=False)
    per_sample.to_parquet(final_path, index=False)

    aggregate_rows = []
    for class_name in CLASSES:
        subset = per_sample[per_sample["class"] == class_name]
        signed = subset[names].mean()
        absolute = subset[names].abs().mean()
        for rank, feature in enumerate(
            absolute.sort_values(ascending=False).index, start=1
        ):
            aggregate_rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "signed_mean": float(signed[feature]),
                    "abs_mean": float(absolute[feature]),
                    "rank": rank,
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate_path = output_dir / "ablation_feature_attribution.csv"
    aggregate.to_csv(aggregate_path, index=False)

    ig = pd.read_csv(ig_dir / "captum_feature_attribution.csv")
    agreement_rows = []
    for class_name in CLASSES:
        ig_class = ig[ig["class"] == class_name].set_index("feature")
        ablation_class = aggregate[
            aggregate["class"] == class_name
        ].set_index("feature")
        shared = ig_class.index.intersection(ablation_class.index)
        correlation = spearmanr(
            ig_class.loc[shared, "abs_mean"],
            ablation_class.loc[shared, "abs_mean"],
        ).correlation
        ig_top5 = set(ig_class.nsmallest(5, "rank").index)
        ablation_top5 = set(ablation_class.nsmallest(5, "rank").index)
        agreement_rows.append(
            {
                "class": class_name,
                "spearman": float(correlation),
                "top5_overlap_count": len(ig_top5 & ablation_top5),
                "ig_top1": ig_class.nsmallest(1, "rank").index[0],
                "ablation_top1": ablation_class.nsmallest(1, "rank").index[0],
            }
        )
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(output_dir / "ig_ablation_agreement.csv", index=False)

    lines = [
        "# Layer-IG and field-masking comparison",
        "",
        "Both analyses use the same fixed cohort. Masking replaces one serialized "
        "field at a time with [PAD] tokens and measures the change in the true-class "
        "logit. Per-sample changes are L1-normalized before aggregation.",
        "",
        "| Class | Spearman | Top-5 overlap | IG top feature | Masking top feature |",
        "|---|---:|---:|---|---|",
    ]
    for _, row in agreement.iterrows():
        lines.append(
            f"| {row['class']} | {row['spearman']:+.3f} | "
            f"{int(row['top5_overlap_count'])}/5 | `{row['ig_top1']}` | "
            f"`{row['ablation_top1']}` |"
        )
    lines.extend(
        [
            "",
            f"Mean Spearman: {agreement['spearman'].mean():+.3f}. Mean top-5 "
            f"overlap: {agreement['top5_overlap_count'].mean():.2f}/5.",
            "",
            "Field masking is a complementary perturbation check, not proof that the "
            "gradient attribution is correct. Agreement strengthens a local reading; "
            "disagreement should be reported rather than averaged away.",
            "",
        ]
    )
    (output_dir / "ablation_crosscheck.md").write_text("\n".join(lines))

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set": args.feature_set,
        "device": str(device),
        "cohort_rows": len(per_sample),
        "feature_count": len(names),
        "inputs": {
            "checkpoint_sha256": sha256(model_dir / "model.safetensors"),
            "serialized_test_sha256": sha256(serialized_path),
            "cohort_sha256": sha256(cohort_path),
            "ig_attribution_sha256": sha256(
                ig_dir / "captum_feature_attribution.csv"
            ),
        },
        "outputs": {
            "per_sample_sha256": sha256(final_path),
            "feature_attribution_sha256": sha256(aggregate_path),
        },
    }
    (output_dir / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("\n".join(lines))
    print(f"wrote revised masking cross-check to {output_dir}")


if __name__ == "__main__":
    main()
