"""Run Layer Integrated Gradients on a fixed revised XAI cohort.

The cohort is supplied explicitly and must contain rows that are correctly
classified by the model being explained. Token attributions are remapped to
the serialized flow fields, L1-normalized per sample, and aggregated by true
class. A progress parquet makes an interrupted attribution run resumable.
"""

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
from captum.attr import LayerIntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline_config import CLASSES


ROOT = Path(__file__).resolve().parents[1]
FIELD_RE = re.compile(r"([A-Za-z_]+)=")
MAX_LEN = 224


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=("original", "no_number"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--serialized-test", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--internal-batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Attribute the first aligned row without writing result artifacts.",
    )
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


def feature_spans(text):
    keys = [(match.group(1), match.start()) for match in FIELD_RE.finditer(text)]
    return [
        (name, start, keys[index + 1][1] if index + 1 < len(keys) else len(text))
        for index, (name, start) in enumerate(keys)
    ]


def token_to_feature(offsets, spans):
    mapping = np.full(len(offsets), -1, dtype=np.int64)
    for token_index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        for feature_index, (_, feature_start, feature_end) in enumerate(spans):
            if feature_start <= start < feature_end:
                mapping[token_index] = feature_index
                break
    return mapping


def aggregate(per_sample: pd.DataFrame, feature_columns) -> pd.DataFrame:
    rows = []
    for class_name in CLASSES:
        subset = per_sample[per_sample["class"] == class_name]
        signed = subset[feature_columns].mean()
        absolute = subset[feature_columns].abs().mean()
        for rank, feature in enumerate(
            absolute.sort_values(ascending=False).index, start=1
        ):
            rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "signed_mean": float(signed[feature]),
                    "abs_mean": float(absolute[feature]),
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if args.n_steps <= 0 or args.internal_batch_size <= 0:
        raise ValueError("IG step and internal-batch values must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")

    model_dir = args.model_dir.resolve()
    serialized_path = args.serialized_test.resolve()
    cohort_path = args.cohort.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "captum_per_sample.parquet"
    progress_path = output_dir / "captum_per_sample_progress.parquet"
    if final_path.exists():
        raise FileExistsError(f"completed output already exists at {final_path}")
    if progress_path.exists() and not args.resume:
        raise FileExistsError(
            f"progress exists at {progress_path}; pass --resume to continue"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    serialized = pd.read_parquet(serialized_path).reset_index(drop=True)
    cohort = pd.read_parquet(cohort_path).sort_values("cohort_order")
    required_serialized = {"SampleID", "Category", "text"}
    required_cohort = {"cohort_order", "row", "SampleID", "true_category", "y_true"}
    if required_serialized - set(serialized.columns):
        raise ValueError("serialized test data are missing required columns")
    if required_cohort - set(cohort.columns):
        raise ValueError("cohort is missing required columns")
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

    # Recheck model correctness on the exact cohort before explaining it.
    predictions = []
    with torch.no_grad():
        for start in range(0, len(selected), 128):
            texts = selected["text"].iloc[start : start + 128].tolist()
            encoded = tokenizer(
                texts,
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            ).to(device)
            predictions.extend(model(**encoded).logits.argmax(1).cpu().tolist())
    if not np.array_equal(np.asarray(predictions), selected["y_true"].to_numpy()):
        raise ValueError("the supplied cohort contains rows not correctly classified")

    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    def forward_fn(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    integrated_gradients = LayerIntegratedGradients(
        forward_fn, model.distilbert.embeddings
    )
    records = (
        pd.read_parquet(progress_path).to_dict("records")
        if progress_path.exists()
        else []
    )
    completed = {record["SampleID"] for record in records}
    pending = selected[~selected["SampleID"].isin(completed)]
    print(
        f"device {device}; feature_set {args.feature_set}; cohort {len(selected)}; "
        f"completed {len(completed)}; pending {len(pending)}; n_steps {args.n_steps}",
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
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        reference = input_ids.clone()
        keep = (input_ids == cls_id) | (input_ids == sep_id)
        reference[~keep] = pad_id

        attributions, convergence_delta = integrated_gradients.attribute(
            inputs=input_ids,
            baselines=reference,
            additional_forward_args=(attention_mask,),
            target=int(row.y_true),
            n_steps=args.n_steps,
            internal_batch_size=args.internal_batch_size,
            return_convergence_delta=True,
        )
        token_attribution = (
            attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        )
        spans = feature_spans(row.text)
        mapping = token_to_feature(offsets, spans)
        names = [span[0] for span in spans]
        feature_attribution = {name: 0.0 for name in names}
        for token_index, feature_index in enumerate(mapping):
            if feature_index >= 0:
                feature_attribution[names[feature_index]] += float(
                    token_attribution[token_index]
                )
        total = sum(abs(value) for value in feature_attribution.values()) or 1.0
        record = {
            "class": row.true_category,
            "row": int(row.row),
            "SampleID": row.SampleID,
            "convergence_delta": float(convergence_delta.detach().cpu().item()),
        }
        record.update(
            {name: value / total for name, value in feature_attribution.items()}
        )
        records.append(record)

        if processed % args.save_every == 0:
            pd.DataFrame(records).fillna(0.0).to_parquet(progress_path, index=False)
            print(
                f"  attributed {len(records)}/{len(selected)} rows",
                flush=True,
            )

    if args.dry_run:
        delta = abs(float(records[0]["convergence_delta"]))
        print(f"dry run passed; one attribution completed; abs delta {delta:.6f}")
        return

    per_sample = pd.DataFrame(records).fillna(0.0)
    if len(per_sample) != len(selected) or not per_sample["SampleID"].is_unique:
        raise ValueError("final attribution rows are incomplete or duplicated")
    per_sample = selected[["cohort_order", "SampleID"]].merge(
        per_sample, on="SampleID", validate="one_to_one"
    ).sort_values("cohort_order")
    per_sample.to_parquet(progress_path, index=False)
    per_sample.to_parquet(final_path, index=False)

    metadata_columns = {
        "cohort_order",
        "class",
        "row",
        "SampleID",
        "convergence_delta",
    }
    feature_columns = [
        column for column in per_sample.columns if column not in metadata_columns
    ]
    aggregated = aggregate(per_sample, feature_columns)
    attribution_path = output_dir / "captum_feature_attribution.csv"
    aggregated.to_csv(attribution_path, index=False)

    absolute_delta = per_sample["convergence_delta"].abs()
    lines = [
        f"# DistilBERT `{args.feature_set}` — Layer Integrated Gradients",
        "",
        f"The analysis uses {len(per_sample)} fixed cohort rows, {args.n_steps} "
        "integration steps, an all-[PAD] content baseline, and per-sample L1 "
        "normalization after mapping wordpieces back to flow fields.",
        "",
        "## Cohort size",
        "",
        per_sample.groupby("class", sort=False).size().rename("rows").reset_index().to_markdown(index=False),
        "",
        "## Top features by class",
        "",
    ]
    for class_name in CLASSES:
        top = aggregated[aggregated["class"] == class_name].nsmallest(8, "rank")
        rendered = ", ".join(
            f"`{row.feature}` ({row.abs_mean:.4f})"
            for row in top.itertuples(index=False)
        )
        lines.append(f"- **{class_name}:** {rendered}")
    lines.extend(
        [
            "",
            "## Completeness diagnostic",
            "",
            f"Absolute convergence delta: mean {absolute_delta.mean():.6f}, "
            f"median {absolute_delta.median():.6f}, maximum {absolute_delta.max():.6f}.",
            "",
            "The convergence delta is retained per sample so attribution quality can "
            "be inspected rather than assumed.",
            "",
        ]
    )
    summary_path = output_dir / "captum_summary.md"
    summary_path.write_text("\n".join(lines))

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set": args.feature_set,
        "device": str(device),
        "seed": args.seed,
        "max_length": MAX_LEN,
        "n_steps": args.n_steps,
        "internal_batch_size": args.internal_batch_size,
        "cohort_rows": len(per_sample),
        "class_counts": {
            str(class_name): int(count)
            for class_name, count in per_sample["class"].value_counts(sort=False).items()
        },
        "feature_count": len(feature_columns),
        "inputs": {
            "checkpoint": {
                "path": str((model_dir / "model.safetensors").relative_to(ROOT)),
                "sha256": sha256(model_dir / "model.safetensors"),
            },
            "serialized_test": {
                "path": str(serialized_path.relative_to(ROOT)),
                "sha256": sha256(serialized_path),
            },
            "cohort": {
                "path": str(cohort_path.relative_to(ROOT)),
                "sha256": sha256(cohort_path),
            },
        },
        "outputs": {
            "per_sample_sha256": sha256(final_path),
            "feature_attribution_sha256": sha256(attribution_path),
        },
        "convergence_delta_abs": {
            "mean": float(absolute_delta.mean()),
            "median": float(absolute_delta.median()),
            "max": float(absolute_delta.max()),
        },
    }
    (output_dir / "captum_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"wrote revised Layer IG artifacts to {output_dir}")


if __name__ == "__main__":
    main()
