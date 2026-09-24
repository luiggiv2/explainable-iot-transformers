"""Add a scale-aware completeness audit to a completed Layer-IG run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MAX_LEN = 224


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--serialized-test", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--ig-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str):
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested, but MPS is unavailable")
    return torch.device(requested)


def main():
    args = parse_args()
    model_dir = args.model_dir.resolve()
    serialized_path = args.serialized_test.resolve()
    cohort_path = args.cohort.resolve()
    ig_dir = args.ig_dir.resolve()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    manifest_path = ig_dir / "captum_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["inputs"]["checkpoint"]["sha256"] != sha256(
        model_dir / "model.safetensors"
    ):
        raise ValueError("checkpoint differs from the completed IG run")
    if manifest["inputs"]["cohort"]["sha256"] != sha256(cohort_path):
        raise ValueError("cohort differs from the completed IG run")

    per_sample = pd.read_parquet(ig_dir / "captum_per_sample.parquet")
    serialized = pd.read_parquet(serialized_path).reset_index(names="serialized_row")
    cohort = pd.read_parquet(cohort_path)
    selected = cohort.merge(
        serialized[["serialized_row", "SampleID", "Category", "text"]],
        left_on=["row", "SampleID", "true_category"],
        right_on=["serialized_row", "SampleID", "Category"],
        validate="one_to_one",
    ).sort_values("cohort_order")
    selected = selected.merge(
        per_sample[["SampleID", "convergence_delta"]],
        on="SampleID",
        validate="one_to_one",
    ).sort_values("cohort_order")
    if len(selected) != manifest["cohort_rows"]:
        raise ValueError("IG records do not cover the full cohort")

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    input_logits = []
    baseline_logits = []
    with torch.no_grad():
        for start in range(0, len(selected), args.batch_size):
            batch = selected.iloc[start : start + args.batch_size]
            encoded = tokenizer(
                batch["text"].tolist(),
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            ).to(device)
            reference = encoded["input_ids"].clone()
            keep = (reference == cls_id) | (reference == sep_id)
            reference[~keep] = pad_id
            targets = torch.tensor(batch["y_true"].to_numpy(), device=device)
            observed = model(**encoded).logits.gather(1, targets[:, None]).squeeze(1)
            baseline = model(
                input_ids=reference, attention_mask=encoded["attention_mask"]
            ).logits.gather(1, targets[:, None]).squeeze(1)
            input_logits.extend(observed.cpu().tolist())
            baseline_logits.extend(baseline.cpu().tolist())

    audit = selected[["cohort_order", "row", "SampleID", "true_category"]].copy()
    audit["input_target_logit"] = input_logits
    audit["baseline_target_logit"] = baseline_logits
    audit["logit_difference"] = audit["input_target_logit"] - audit["baseline_target_logit"]
    audit["convergence_delta"] = selected["convergence_delta"].to_numpy()
    audit["absolute_delta"] = audit["convergence_delta"].abs()
    audit["relative_delta"] = audit["absolute_delta"] / audit[
        "logit_difference"
    ].abs().clip(lower=1e-6)
    audit_path = ig_dir / "captum_completeness_audit.csv"
    audit.to_csv(audit_path, index=False)

    absolute = audit["absolute_delta"]
    relative = audit["relative_delta"]
    lines = [
        "# Layer-IG completeness audit",
        "",
        "Captum's convergence residual is reported both in raw logit units and "
        "relative to the absolute target-logit difference between the observed "
        "input and the all-[PAD] content baseline.",
        "",
        f"- absolute residual: median {absolute.median():.6f}, mean "
        f"{absolute.mean():.6f}, 95th percentile {absolute.quantile(0.95):.6f}, "
        f"maximum {absolute.max():.6f};",
        f"- relative residual: median {relative.median():.2%}, mean "
        f"{relative.mean():.2%}, 95th percentile {relative.quantile(0.95):.2%}, "
        f"maximum {relative.max():.2%};",
        f"- rows at or below 5% relative residual: {(relative <= 0.05).mean():.1%};",
        f"- rows at or below 10% relative residual: {(relative <= 0.10).mean():.1%}.",
        "",
        "A large relative value can occur when the reference-to-input logit "
        "difference is close to zero, so the per-row audit should be consulted "
        "before deciding whether additional integration steps are warranted.",
        "",
    ]
    summary_path = ig_dir / "captum_completeness_audit.md"
    summary_path.write_text("\n".join(lines))
    manifest["quality_audit"] = {
        "device": str(device),
        "relative_delta_median": float(relative.median()),
        "relative_delta_p95": float(relative.quantile(0.95)),
        "relative_delta_max": float(relative.max()),
        "fraction_relative_delta_le_0_05": float((relative <= 0.05).mean()),
        "fraction_relative_delta_le_0_10": float((relative <= 0.10).mean()),
        "audit_csv_sha256": sha256(audit_path),
        "audit_summary_sha256": sha256(summary_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
