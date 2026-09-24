"""Revised explanation-robustness analysis for the representative encoder.

The script selects a fixed, class-balanced cohort from correctly classified
test rows, applies deterministic bounded multiplicative perturbations, and
compares Layer Integrated Gradients maps before and after perturbation. All
outputs are written below one run directory and are resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline_config import CLASSES, feature_pairs, format_value


FIELD_RE = re.compile(r"([A-Za-z_]+)=")
FRACTION_FIELDS = {
    "fin", "syn", "rst", "psh", "ack", "ece", "cwr", "http", "https",
    "dns", "telnet", "smtp", "ssh", "irc", "tcp", "udp", "dhcp", "arp",
    "icmp", "igmp", "ipv", "llc",
}
COUNT_FIELDS = {"ack_cnt", "syn_cnt", "fin_cnt", "rst_cnt", "num"}
MAX_LEN = 224


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--serialized-test", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-per-class", type=int, default=60)
    parser.add_argument("--eps", type=float, nargs="+", default=(0.01, 0.05, 0.10))
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--internal-batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
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


def stable_rng(seed: int, sample_id: str, epsilon: float):
    material = f"{seed}|{sample_id}|{epsilon:.12g}".encode()
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(derived)


def serialize(values, names):
    return " ".join(
        f"{name}={format_value(value)}" for name, value in zip(names, values)
    )


def perturb(values, names, epsilon, rng):
    changed = values.astype(np.float64).copy()
    changed *= 1.0 + epsilon * rng.standard_normal(len(changed))
    for index, name in enumerate(names):
        if name == "proto_num":
            changed[index] = values[index]
        elif name in FRACTION_FIELDS:
            changed[index] = np.clip(changed[index], 0.0, 1.0)
        elif name in COUNT_FIELDS:
            changed[index] = max(round(changed[index]), 0.0)
        else:
            changed[index] = max(changed[index], 0.0)
    return changed


def token_field_map(text, offsets, expected_names):
    matches = [(match.group(1), match.start()) for match in FIELD_RE.finditer(text)]
    names = [name for name, _ in matches]
    if names != expected_names:
        raise ValueError("serialized field names or order do not match the protocol")
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


def exact_permutation_p(x, y):
    centered_x = np.asarray(x, dtype=np.float64) - np.mean(x)
    centered_y = np.asarray(y, dtype=np.float64) - np.mean(y)
    denominator = np.linalg.norm(centered_x) * np.linalg.norm(centered_y)
    if denominator == 0:
        raise ValueError("class-level robustness values are constant")
    observed = abs(float(centered_x @ centered_y / denominator))
    permutations = np.asarray(list(itertools.permutations(centered_y)))
    statistics = np.abs(permutations @ centered_x / denominator)
    return observed, float(np.mean(statistics >= observed - 1e-12))


def main():
    args = parse_args()
    if args.n_per_class <= 0 or args.n_steps <= 0 or args.save_every <= 0:
        raise ValueError("sample, integration-step, and checkpoint values must be positive")
    if not args.eps or any(level <= 0 for level in args.eps):
        raise ValueError("all perturbation levels must be positive")

    model_dir = args.model_dir.resolve()
    split_path = args.test_split.resolve()
    serialized_path = args.serialized_test.resolve()
    predictions_path = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = output_dir / "robustness_cohort.parquet"
    progress_path = output_dir / "robustness_records_progress.parquet"
    final_path = output_dir / "robustness_records.parquet"
    if final_path.exists():
        raise FileExistsError(f"completed output already exists at {final_path}")
    if progress_path.exists() and not args.resume:
        raise FileExistsError(f"progress exists at {progress_path}; pass --resume")

    pairs = feature_pairs("original")
    columns = [column for column, _ in pairs]
    names = [name for _, name in pairs]
    split = pd.read_parquet(split_path).reset_index(names="row")
    serialized = pd.read_parquet(serialized_path).reset_index(names="serialized_row")
    predictions = pd.read_parquet(predictions_path)
    aligned = split.merge(
        serialized[["serialized_row", "SampleID", "Category", "text"]],
        left_on=["row", "SampleID", "Category"],
        right_on=["serialized_row", "SampleID", "Category"],
        validate="one_to_one",
    ).merge(
        predictions[["row", "SampleID", "true_category", "y_true", "y_pred"]],
        left_on=["row", "SampleID", "Category"],
        right_on=["row", "SampleID", "true_category"],
        validate="one_to_one",
    )
    if len(aligned) != len(split):
        raise ValueError("split, serialization, and predictions did not align completely")

    if cohort_path.exists():
        cohort = pd.read_parquet(cohort_path)
    else:
        rng = np.random.default_rng(args.seed)
        chosen = []
        correct = aligned[aligned["y_true"] == aligned["y_pred"]]
        for class_name in CLASSES:
            pool = correct[correct["Category"] == class_name]
            if pool.empty:
                raise ValueError(f"no correctly classified rows for {class_name}")
            count = min(args.n_per_class, len(pool))
            indices = rng.choice(pool.index.to_numpy(), count, replace=False)
            chosen.append(aligned.loc[indices])
        cohort = pd.concat(chosen).reset_index(drop=True)
        cohort.insert(0, "cohort_order", np.arange(len(cohort), dtype=np.int64))
        cohort[["cohort_order", "row", "SampleID", "Category", "y_true"]].to_parquet(
            cohort_path, index=False
        )
        cohort = pd.read_parquet(cohort_path)

    expected_counts = cohort.groupby("Category").size().to_dict()
    for class_name in CLASSES:
        available = int(
            ((aligned["Category"] == class_name) & (aligned["y_true"] == aligned["y_pred"])).sum()
        )
        if expected_counts.get(class_name) != min(args.n_per_class, available):
            raise ValueError("saved robustness cohort does not match the requested design")
    selected = cohort.merge(
        aligned,
        on=["row", "SampleID", "Category", "y_true"],
        validate="one_to_one",
    ).sort_values("cohort_order")
    if args.dry_run:
        selected = selected.head(1)

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    def forward_fn(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    integrated_gradients = LayerIntegratedGradients(
        forward_fn, model.distilbert.embeddings
    )

    @torch.no_grad()
    def predict(text):
        encoded = tokenizer(
            text, truncation=True, max_length=MAX_LEN, return_tensors="pt"
        ).to(device)
        return int(model(**encoded).logits.argmax(1).item())

    def attribute(text, target):
        encoded = tokenizer(
            text,
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
        attributions = integrated_gradients.attribute(
            inputs=input_ids,
            baselines=reference,
            additional_forward_args=(attention_mask,),
            target=int(target),
            n_steps=args.n_steps,
            internal_batch_size=args.internal_batch_size,
        )
        token_values = attributions.sum(-1).squeeze(0).detach().cpu().numpy()
        mapping = token_field_map(text, offsets, names)
        values = np.zeros(len(names), dtype=np.float64)
        for token_index, feature_index in enumerate(mapping):
            if feature_index >= 0:
                values[feature_index] += float(token_values[token_index])
        absolute = np.abs(values)
        return absolute / (absolute.sum() or 1.0)

    records = (
        pd.read_parquet(progress_path).to_dict("records")
        if progress_path.exists()
        else []
    )
    completed_counts = pd.Series(
        [record["SampleID"] for record in records], dtype="object"
    ).value_counts()
    malformed = completed_counts[completed_counts != len(args.eps)]
    if not malformed.empty:
        raise ValueError("progress file contains an incomplete sample; remove or repair it")
    completed = set(completed_counts.index)
    pending = selected[~selected["SampleID"].isin(completed)]
    print(
        f"device {device}; cohort {len(selected)}; completed {len(completed)}; "
        f"pending {len(pending)}; eps {args.eps}; n_steps {args.n_steps}",
        flush=True,
    )

    processed_since_save = 0
    for row in pending.itertuples(index=False):
        # itertuples cannot index columns containing spaces; recover the row once by id.
        numeric = aligned.loc[aligned["SampleID"] == row.SampleID, columns].iloc[0].to_numpy(np.float64)
        base_text = row.text
        if predict(base_text) != int(row.y_true):
            raise ValueError(f"cohort row is no longer correctly classified: {row.SampleID}")
        base_attr = attribute(base_text, int(row.y_true))
        base_top5 = set(np.argsort(base_attr)[::-1][:5])
        sample_records = []
        for epsilon in args.eps:
            changed = perturb(
                numeric, names, epsilon, stable_rng(args.seed, row.SampleID, epsilon)
            )
            changed_text = serialize(changed, names)
            changed_attr = attribute(changed_text, int(row.y_true))
            changed_top5 = set(np.argsort(changed_attr)[::-1][:5])
            correlation = spearmanr(base_attr, changed_attr).correlation
            intersection = len(base_top5 & changed_top5)
            sample_records.append(
                {
                    "cohort_order": int(row.cohort_order),
                    "row": int(row.row),
                    "SampleID": row.SampleID,
                    "class": row.Category,
                    "eps": float(epsilon),
                    "pred_preserved": predict(changed_text) == int(row.y_true),
                    "spearman": float(correlation) if np.isfinite(correlation) else np.nan,
                    "top5_overlap_count": intersection,
                    "top5_jaccard": intersection / (10 - intersection),
                }
            )
        records.extend(sample_records)
        processed_since_save += 1
        if processed_since_save % args.save_every == 0:
            pd.DataFrame(records).to_parquet(progress_path, index=False)
            print(
                f"  robustness {len(set(r['SampleID'] for r in records))}/{len(selected)} rows",
                flush=True,
            )
            if device.type == "mps":
                torch.mps.empty_cache()

    result = pd.DataFrame(records)
    expected_rows = len(selected) * len(args.eps)
    if len(result) != expected_rows:
        raise ValueError(f"expected {expected_rows} result rows, found {len(result)}")
    if result.duplicated(["SampleID", "eps"]).any():
        raise ValueError("duplicate SampleID/eps rows in robustness output")
    result = result.sort_values(["cohort_order", "eps"])
    if args.dry_run:
        print(result.to_string(index=False))
        return
    result.to_parquet(progress_path, index=False)
    result.to_parquet(final_path, index=False)

    metric_rows = []
    for class_name in CLASSES + ["_ALL"]:
        for epsilon in args.eps:
            subset = result[result["eps"] == epsilon]
            if class_name != "_ALL":
                subset = subset[subset["class"] == class_name]
            metric_rows.append(
                {
                    "class": class_name,
                    "eps": epsilon,
                    "n": len(subset),
                    "pred_preserved_frac": subset["pred_preserved"].mean(),
                    "mean_spearman": subset["spearman"].mean(),
                    "mean_top5_overlap": subset["top5_overlap_count"].mean(),
                    "mean_top5_jaccard": subset["top5_jaccard"].mean(),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "robustness_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    per_class = (
        result.groupby("class", as_index=False)
        .agg(
            pred_preserved=("pred_preserved", "mean"),
            top5_overlap=("top5_overlap_count", "mean"),
            top5_jaccard=("top5_jaccard", "mean"),
            spearman=("spearman", "mean"),
        )
        .set_index("class")
        .loc[CLASSES]
        .reset_index()
    )
    correlation, permutation_p = exact_permutation_p(
        per_class["pred_preserved"].to_numpy(),
        per_class["top5_overlap"].to_numpy(),
    )
    per_class_path = output_dir / "robustness_per_class.csv"
    per_class.to_csv(per_class_path, index=False)

    lines = [
        "# Revised explanation robustness analysis",
        "",
        f"The analysis uses {len(selected)} correctly classified test flows "
        f"({args.n_per_class} per class where available), with deterministic "
        "multiplicative perturbations of 1%, 5%, and 10%. Protocol identity and "
        "zero-valued indicator support remain unchanged.",
        "",
        "## Overall results",
        "",
        "| Perturbation | Prediction preserved | Mean Spearman | Mean top-5 overlap |",
        "|---:|---:|---:|---:|",
    ]
    for _, row in metrics[metrics["class"] == "_ALL"].iterrows():
        lines.append(
            f"| {row['eps']:.0%} | {row['pred_preserved_frac']:.1%} | "
            f"{row['mean_spearman']:.3f} | {row['mean_top5_overlap']:.2f}/5 |"
        )
    lines.extend(
        [
            "",
            "## Results by class, pooled across perturbation levels",
            "",
            "| Class | Prediction preserved | Mean Spearman | Mean top-5 overlap |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in per_class.iterrows():
        lines.append(
            f"| {row['class']} | {row['pred_preserved']:.1%} | "
            f"{row['spearman']:.3f} | {row['top5_overlap']:.2f}/5 |"
        )
    lines.extend(
        [
            "",
            "Across the eight classes, prediction preservation and mean top-5 "
            f"overlap have Pearson r = {correlation:+.3f} (two-sided exact "
            f"permutation p = {permutation_p:.4f}). With only eight class-level "
            "observations, this association is descriptive and should not be read "
            "as a broadly generalizable estimate.",
            "",
            "The perturbations probe local sensitivity within this representation. "
            "They are not adversarial attacks and do not establish robustness under "
            "distribution shift.",
            "",
        ]
    )
    summary_path = output_dir / "robustness_summary.md"
    summary_path.write_text("\n".join(lines))

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "seed": args.seed,
        "n_per_class": args.n_per_class,
        "eps": list(args.eps),
        "n_steps": args.n_steps,
        "internal_batch_size": args.internal_batch_size,
        "inputs": {
            "checkpoint_sha256": sha256(model_dir / "model.safetensors"),
            "test_split_sha256": sha256(split_path),
            "serialized_test_sha256": sha256(serialized_path),
            "predictions_sha256": sha256(predictions_path),
            "cohort_sha256": sha256(cohort_path),
        },
        "outputs": {
            "records_sha256": sha256(final_path),
            "metrics_sha256": sha256(metrics_path),
            "per_class_sha256": sha256(per_class_path),
            "summary_sha256": sha256(summary_path),
        },
    }
    (output_dir / "robustness_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
