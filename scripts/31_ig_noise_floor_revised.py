"""Revised IG noise floor (eps=0 control) for the RQ4 robustness metrics.

The robustness analysis reports how much a Layer-IG map moves when the input is
perturbed. That number is only interpretable against a floor: how much the same
map moves when the input is *not* perturbed at all. This control attributes each
unperturbed input twice and applies the identical aggregation, normalization and
metrics used by `28_robustness_revised.py`, on a deterministic subsample of that
same cohort, on the same device.

A floor at Spearman 1.000 and 5/5 top-5 overlap licenses the claim that the
instability reported for RQ4 is input sensitivity rather than method noise. Any
departure from that must be reported and subtracted from the interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline_config import CLASSES, feature_pairs


MAX_LEN = 224


def _load_robustness_module():
    """Reuse the token/field mapping of the robustness script verbatim.

    Importing it rather than re-implementing it guarantees the floor and the
    perturbation study aggregate token attributions to fields identically.
    """
    path = Path(__file__).resolve().parent / "28_robustness_revised.py"
    spec = importlib.util.spec_from_file_location("robustness_revised", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--serialized-test", type=Path, required=True)
    parser.add_argument("--robustness-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-per-class", type=int, default=15)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--internal-batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.n_per_class <= 0 or args.n_steps <= 0:
        raise ValueError("sample and integration-step counts must be positive")

    robustness = _load_robustness_module()
    if robustness.MAX_LEN != MAX_LEN:
        raise ValueError("maximum sequence length drifted from the robustness study")

    model_dir = args.model_dir.resolve()
    robustness_dir = args.robustness_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "noise_floor_per_sample.parquet"
    if final_path.exists():
        raise FileExistsError(f"completed output already exists at {final_path}")

    cohort_path = robustness_dir / "robustness_cohort.parquet"
    cohort = pd.read_parquet(cohort_path)

    pairs = feature_pairs("original")
    names = [name for _, name in pairs]
    split = pd.read_parquet(args.test_split).reset_index(names="row")
    serialized = pd.read_parquet(args.serialized_test).reset_index(
        names="serialized_row"
    )
    aligned = split.merge(
        serialized[["serialized_row", "SampleID", "Category", "text"]],
        left_on=["row", "SampleID", "Category"],
        right_on=["serialized_row", "SampleID", "Category"],
        validate="one_to_one",
    )

    # Deterministic subsample of the robustness cohort: the floor must be
    # measured on the rows whose perturbed behaviour it is used to interpret.
    rng = np.random.default_rng(args.seed)
    chosen = []
    for class_name in CLASSES:
        pool = cohort[cohort["Category"] == class_name]
        if pool.empty:
            raise ValueError(f"robustness cohort has no rows for {class_name}")
        count = min(args.n_per_class, len(pool))
        indices = rng.choice(pool.index.to_numpy(), count, replace=False)
        chosen.append(cohort.loc[indices])
    selected = (
        pd.concat(chosen)
        .merge(aligned, on=["row", "SampleID", "Category"], validate="one_to_one")
        .sort_values("cohort_order")
    )
    if args.dry_run:
        selected = selected.head(2)

    torch.manual_seed(args.seed)
    device = robustness.resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    integrated_gradients = LayerIntegratedGradients(
        lambda input_ids, attention_mask: model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits,
        model.distilbert.embeddings,
    )

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
        mapping = robustness.token_field_map(text, offsets, names)
        values = np.zeros(len(names), dtype=np.float64)
        for token_index, feature_index in enumerate(mapping):
            if feature_index >= 0:
                values[feature_index] += float(token_values[token_index])
        absolute = np.abs(values)
        return absolute / (absolute.sum() or 1.0)

    records = []
    print(
        f"device {device}; rows {len(selected)}; n_steps {args.n_steps}",
        flush=True,
    )
    for position, row in enumerate(selected.itertuples(index=False), start=1):
        first = attribute(row.text, int(row.y_true))
        second = attribute(row.text, int(row.y_true))
        first_top5 = set(np.argsort(first)[::-1][:5])
        second_top5 = set(np.argsort(second)[::-1][:5])
        top8 = np.argsort(first)[::-1][:8]
        top8_correlation = spearmanr(first[top8], second[top8]).correlation
        correlation = spearmanr(first, second).correlation
        records.append(
            {
                "cohort_order": int(row.cohort_order),
                "row": int(row.row),
                "SampleID": row.SampleID,
                "class": row.Category,
                "spearman": float(correlation) if np.isfinite(correlation) else 1.0,
                "spearman_top8": float(top8_correlation)
                if np.isfinite(top8_correlation)
                else 1.0,
                "top5_overlap_count": len(first_top5 & second_top5),
                "max_abs_difference": float(np.max(np.abs(first - second))),
                "identical": bool(np.array_equal(first, second)),
            }
        )
        if position % 20 == 0:
            print(f"  noise floor {position}/{len(selected)} rows", flush=True)

    result = pd.DataFrame(records).sort_values("cohort_order")
    if args.dry_run:
        print(result.to_string(index=False))
        return
    result.to_parquet(final_path, index=False)

    per_class = (
        result.groupby("class", as_index=False)
        .agg(
            n=("SampleID", "size"),
            spearman=("spearman", "mean"),
            spearman_top8=("spearman_top8", "mean"),
            top5_overlap=("top5_overlap_count", "mean"),
            max_abs_difference=("max_abs_difference", "max"),
            identical_fraction=("identical", "mean"),
        )
        .set_index("class")
        .loc[CLASSES]
        .reset_index()
    )
    per_class_path = output_dir / "noise_floor_per_class.csv"
    per_class.to_csv(per_class_path, index=False)

    floor_spearman = result["spearman"].mean()
    floor_overlap = result["top5_overlap_count"].mean()
    deterministic = bool(result["identical"].all())

    lines = [
        "# Layer-IG noise floor (eps = 0 control)",
        "",
        f"Each of the {len(result)} inputs is drawn from the robustness cohort and "
        "attributed twice without any perturbation, using the same integration "
        "settings, field aggregation, normalization and metrics as the "
        "perturbation study, on the same device.",
        "",
        "## Floor",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean Spearman over all {len(names)} fields | {floor_spearman:.4f} |",
        f"| Mean Spearman over the top-8 fields | {result['spearman_top8'].mean():.4f} |",
        f"| Mean top-5 overlap | {floor_overlap:.2f}/5 |",
        f"| Bitwise-identical attribution vectors | {100 * result['identical'].mean():.1f}% |",
        f"| Largest absolute difference in any field | {result['max_abs_difference'].max():.3e} |",
        "",
    ]

    metrics_path = robustness_dir / "robustness_metrics.csv"
    if metrics_path.exists():
        observed = pd.read_csv(metrics_path)
        observed = observed[observed["class"] == "_ALL"].sort_values("eps")
        lines.extend(
            [
                "## Floor compared with the perturbed measurements",
                "",
                "| Condition | Mean Spearman | Mean top-5 overlap |",
                "|---|---:|---:|",
                f"| eps = 0 (this control) | {floor_spearman:.3f} | {floor_overlap:.2f}/5 |",
            ]
        )
        for _, row in observed.iterrows():
            lines.append(
                f"| eps = {row['eps']:.0%} | {row['mean_spearman']:.3f} | "
                f"{row['mean_top5_overlap']:.2f}/5 |"
            )
        lines.append("")

    if deterministic:
        lines.append(
            "Repeated attribution of an unperturbed input reproduces the field "
            "vector exactly, so the floor is degenerate: every departure from it "
            "in the perturbation study is attributable to the changed input, not "
            "to the attribution method. This does not make individual rankings "
            "reliable — it only removes method noise as an explanation for their "
            "movement."
        )
    else:
        lines.append(
            "Repeated attribution of an unperturbed input does **not** reproduce "
            "the field vector exactly. The residual movement reported above is a "
            "floor that must be subtracted from the interpretation of the "
            "perturbation results, and rank changes of comparable size cannot be "
            "attributed to input sensitivity."
        )
    lines.append("")

    summary_path = output_dir / "noise_floor_summary.md"
    summary_path.write_text("\n".join(lines))

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "seed": args.seed,
        "n_per_class": args.n_per_class,
        "n_steps": args.n_steps,
        "internal_batch_size": args.internal_batch_size,
        "rows": len(result),
        "feature_count": len(names),
        "deterministic": deterministic,
        "floor": {
            "mean_spearman": floor_spearman,
            "mean_spearman_top8": float(result["spearman_top8"].mean()),
            "mean_top5_overlap": floor_overlap,
            "max_abs_difference": float(result["max_abs_difference"].max()),
        },
        "inputs": {
            "checkpoint_sha256": sha256(model_dir / "model.safetensors"),
            "test_split_sha256": sha256(args.test_split.resolve()),
            "serialized_test_sha256": sha256(args.serialized_test.resolve()),
            "robustness_cohort_sha256": sha256(cohort_path),
        },
        "outputs": {
            "per_sample_sha256": sha256(final_path),
            "per_class_sha256": sha256(per_class_path),
            "summary_sha256": sha256(summary_path),
        },
    }
    (output_dir / "noise_floor_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
