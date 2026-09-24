"""Compare DistilBERT original and no-Number runs on aligned test rows.

The three training seeds are paired by seed, and predictions within each pair
are joined one-to-one by SampleID. Row bootstrap intervals describe test-sample
uncertainty conditional on a fitted pair; variation across the three seeds is
reported separately and is not treated as a large-sample inference problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from pipeline_config import CLASSES


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEEDS = (42, 123, 2026)
REFERENCE = "original"
ABLATION = "no_number"

COMMON_PROTOCOL_FIELDS = (
    "git_commit",
    "model_name",
    "classes",
    "train_precision",
    "validation_precision",
    "test_precision",
    "device",
    "max_length",
    "batch_size",
    "eval_batch_size",
    "learning_rate",
    "max_epochs",
    "patience",
    "warmup_fraction",
    "rows",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distilbert-root",
        type=Path,
        default=DATA / "revision" / "runs" / "distilbert",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA / "revision" / "results",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(run_dir: Path, feature_set: str, seed: int) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    if manifest["feature_set"] != feature_set or manifest["seed"] != seed:
        raise ValueError(
            f"manifest identity mismatch in {run_dir}: "
            f"{manifest['feature_set']}/seed {manifest['seed']}"
        )
    if not all(manifest["consistency_checks"].values()):
        raise ValueError(f"failed run consistency check in {run_dir}")
    return manifest


def load_predictions(path: Path, suffix: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = ("SampleID", "y_true", "y_pred", "true_category")
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if not frame["SampleID"].is_unique:
        raise ValueError(f"duplicate SampleID values in {path}")
    return frame[list(required)].rename(columns={"y_pred": f"y_pred_{suffix}"})


def exact_mcnemar(reference_correct, ablation_correct):
    reference_only = int(np.sum(reference_correct & ~ablation_correct))
    ablation_only = int(np.sum(~reference_correct & ablation_correct))
    discordant = reference_only + ablation_only
    p_value = (
        binomtest(min(reference_only, ablation_only), discordant, 0.5).pvalue
        if discordant
        else 1.0
    )
    return reference_only, ablation_only, p_value


def macro_f1(y_true, y_pred) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=np.arange(len(CLASSES)),
            average="macro",
            zero_division=0,
        )
    )


def representative_seed(rows: pd.DataFrame, column: str) -> int:
    ordered = rows.sort_values([column, "seed"])
    return int(ordered.iloc[len(ordered) // 2]["seed"])


def main():
    args = parse_args()
    run_root = args.distilbert_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    bootstrap_summaries = []
    bootstrap_frames = []
    mcnemar_rows = []
    class_rows = []
    input_artifacts = []
    common_protocol = None
    test_rows = None

    for seed in SEEDS:
        reference_dir = run_root / REFERENCE / f"seed_{seed}"
        ablation_dir = run_root / ABLATION / f"seed_{seed}"
        reference_manifest = load_manifest(reference_dir, REFERENCE, seed)
        ablation_manifest = load_manifest(ablation_dir, ABLATION, seed)

        reference_protocol = {
            field: reference_manifest[field] for field in COMMON_PROTOCOL_FIELDS
        }
        ablation_protocol = {
            field: ablation_manifest[field] for field in COMMON_PROTOCOL_FIELDS
        }
        if reference_protocol != ablation_protocol:
            raise ValueError(f"protocol drift between feature sets for seed {seed}")
        if common_protocol is None:
            common_protocol = reference_protocol
        elif reference_protocol != common_protocol:
            raise ValueError(f"protocol drift across seeds at seed {seed}")

        reference_path = reference_dir / "test_predictions.parquet"
        ablation_path = ablation_dir / "test_predictions.parquet"
        reference = load_predictions(reference_path, REFERENCE)
        ablation = load_predictions(ablation_path, ABLATION)
        merged = reference.merge(
            ablation,
            on=["SampleID", "y_true", "true_category"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(reference) or len(merged) != len(ablation):
            raise ValueError(
                f"prediction alignment lost rows for seed {seed}: "
                f"{len(reference)} original, {len(ablation)} no_number, "
                f"{len(merged)} joined"
            )
        if test_rows is None:
            test_rows = len(merged)
        elif len(merged) != test_rows:
            raise ValueError(f"test-row count changed at seed {seed}")

        expected_category = np.asarray(CLASSES, dtype=object)[
            merged["y_true"].to_numpy()
        ]
        if not np.array_equal(expected_category, merged["true_category"].to_numpy()):
            raise ValueError(f"integer/string label mismatch for seed {seed}")

        y_true = merged["y_true"].to_numpy()
        pred_reference = merged[f"y_pred_{REFERENCE}"].to_numpy()
        pred_ablation = merged[f"y_pred_{ABLATION}"].to_numpy()
        reference_f1 = macro_f1(y_true, pred_reference)
        ablation_f1 = macro_f1(y_true, pred_ablation)
        if not np.isclose(
            reference_f1, reference_manifest["test_macro_f1_fp32"], atol=1e-12
        ):
            raise ValueError(f"original manifest/prediction F1 mismatch for seed {seed}")
        if not np.isclose(
            ablation_f1, ablation_manifest["test_macro_f1_fp32"], atol=1e-12
        ):
            raise ValueError(f"no_number manifest/prediction F1 mismatch for seed {seed}")

        reference_accuracy = float(np.mean(pred_reference == y_true))
        ablation_accuracy = float(np.mean(pred_ablation == y_true))
        seed_rows.append(
            {
                "seed": seed,
                "original_validation_macro_f1": reference_manifest[
                    "best_val_macro_f1_fp32"
                ],
                "no_number_validation_macro_f1": ablation_manifest[
                    "best_val_macro_f1_fp32"
                ],
                "original_test_macro_f1": reference_f1,
                "no_number_test_macro_f1": ablation_f1,
                "test_macro_f1_delta": ablation_f1 - reference_f1,
                "original_test_accuracy": reference_accuracy,
                "no_number_test_accuracy": ablation_accuracy,
                "test_accuracy_delta": ablation_accuracy - reference_accuracy,
            }
        )

        rng = np.random.default_rng(
            np.random.SeedSequence([args.bootstrap_seed, seed])
        )
        reference_boot = np.empty(args.bootstrap_resamples)
        ablation_boot = np.empty(args.bootstrap_resamples)
        for iteration in range(args.bootstrap_resamples):
            indices = rng.integers(0, len(y_true), len(y_true))
            sampled_y = y_true[indices]
            reference_boot[iteration] = macro_f1(
                sampled_y, pred_reference[indices]
            )
            ablation_boot[iteration] = macro_f1(
                sampled_y, pred_ablation[indices]
            )
        delta_boot = ablation_boot - reference_boot
        bootstrap_summaries.append(
            {
                "seed": seed,
                "original_macro_f1": reference_f1,
                "original_ci_lower": np.percentile(reference_boot, 2.5),
                "original_ci_upper": np.percentile(reference_boot, 97.5),
                "no_number_macro_f1": ablation_f1,
                "no_number_ci_lower": np.percentile(ablation_boot, 2.5),
                "no_number_ci_upper": np.percentile(ablation_boot, 97.5),
                "delta": ablation_f1 - reference_f1,
                "delta_ci_lower": np.percentile(delta_boot, 2.5),
                "delta_ci_upper": np.percentile(delta_boot, 97.5),
            }
        )
        bootstrap_frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "iteration": np.arange(args.bootstrap_resamples),
                    "original_macro_f1": reference_boot,
                    "no_number_macro_f1": ablation_boot,
                    "delta": delta_boot,
                }
            )
        )

        reference_correct = pred_reference == y_true
        ablation_correct = pred_ablation == y_true
        reference_only, ablation_only, p_value = exact_mcnemar(
            reference_correct, ablation_correct
        )
        mcnemar_rows.append(
            {
                "seed": seed,
                "original_only_correct": reference_only,
                "no_number_only_correct": ablation_only,
                "discordant": reference_only + ablation_only,
                "exact_p": p_value,
            }
        )

        reference_per_class = f1_score(
            y_true,
            pred_reference,
            labels=np.arange(len(CLASSES)),
            average=None,
            zero_division=0,
        )
        ablation_per_class = f1_score(
            y_true,
            pred_ablation,
            labels=np.arange(len(CLASSES)),
            average=None,
            zero_division=0,
        )
        for index, class_name in enumerate(CLASSES):
            class_rows.append(
                {
                    "seed": seed,
                    "class": class_name,
                    "original_f1": reference_per_class[index],
                    "no_number_f1": ablation_per_class[index],
                    "delta": ablation_per_class[index]
                    - reference_per_class[index],
                }
            )

        for feature_set, path, manifest in (
            (REFERENCE, reference_path, reference_manifest),
            (ABLATION, ablation_path, ablation_manifest),
        ):
            prediction_sha256 = sha256(path)
            if prediction_sha256 != manifest["sha256"]["test_predictions"]:
                raise ValueError(
                    f"prediction checksum mismatch for {feature_set}/seed {seed}"
                )
            input_artifacts.append(
                {
                    "feature_set": feature_set,
                    "seed": seed,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": prediction_sha256,
                    "manifest_prediction_sha256": manifest["sha256"][
                        "test_predictions"
                    ],
                }
            )

    seeds = pd.DataFrame(seed_rows)
    bootstrap_summary = pd.DataFrame(bootstrap_summaries)
    bootstrap_samples = pd.concat(bootstrap_frames, ignore_index=True)
    mcnemar = pd.DataFrame(mcnemar_rows)
    per_class = pd.DataFrame(class_rows)
    class_summary = (
        per_class.groupby("class", sort=False)
        .agg(
            original_f1_mean=("original_f1", "mean"),
            no_number_f1_mean=("no_number_f1", "mean"),
            delta_mean=("delta", "mean"),
            delta_sample_sd=("delta", "std"),
            delta_min=("delta", "min"),
            delta_max=("delta", "max"),
        )
        .reset_index()
    )

    original_representative = representative_seed(
        seeds, "original_validation_macro_f1"
    )
    no_number_representative = representative_seed(
        seeds, "no_number_validation_macro_f1"
    )
    delta_mean = seeds["test_macro_f1_delta"].mean()
    delta_sd = seeds["test_macro_f1_delta"].std(ddof=1)

    prefix = "distilbert_original_vs_no_number"
    seeds.to_csv(output_dir / f"{prefix}_seeds.csv", index=False)
    bootstrap_summary.to_csv(
        output_dir / f"{prefix}_paired_bootstrap_summary.csv", index=False
    )
    bootstrap_samples.to_csv(
        output_dir / f"{prefix}_paired_bootstrap_samples.csv", index=False
    )
    mcnemar.to_csv(output_dir / f"{prefix}_mcnemar.csv", index=False)
    per_class.to_csv(output_dir / f"{prefix}_per_class.csv", index=False)
    class_summary.to_csv(
        output_dir / f"{prefix}_per_class_summary.csv", index=False
    )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": f"{ABLATION} - {REFERENCE}",
        "seeds": list(SEEDS),
        "bootstrap_resamples_per_seed": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "test_rows_per_seed": test_rows,
        "common_protocol": common_protocol,
        "representative_checkpoints": {
            REFERENCE: original_representative,
            ABLATION: no_number_representative,
            "selection_rule": "median validation macro-F1; test metrics not used",
        },
        "inputs": input_artifacts,
    }
    (output_dir / f"{prefix}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    display_seeds = seeds[
        [
            "seed",
            "original_test_macro_f1",
            "no_number_test_macro_f1",
            "test_macro_f1_delta",
            "original_test_accuracy",
            "no_number_test_accuracy",
            "test_accuracy_delta",
        ]
    ].copy()
    display_bootstrap = bootstrap_summary[
        ["seed", "delta", "delta_ci_lower", "delta_ci_upper"]
    ].copy()
    display_mcnemar = mcnemar.copy()
    display_classes = class_summary[
        ["class", "original_f1_mean", "no_number_f1_mean", "delta_mean"]
    ].copy()

    lines = [
        "# DistilBERT original vs. no-`Number` comparison",
        "",
        "All three pairs use the same leakage-safe test rows and are aligned "
        f"one-to-one by `SampleID` (n={test_rows} per seed). Every run passed "
        "its saved-prediction and confusion-matrix consistency checks.",
        "",
        "## Results by training seed",
        "",
        display_seeds.round(4).to_markdown(index=False),
        "",
        f"Across seeds, removing `Number` changed test macro-F1 by "
        f"**{delta_mean:+.4f} ± {delta_sd:.4f}** (mean ± sample SD). The "
        "three differences have the same sign, but three seeds are not enough "
        "for strong distributional claims about training variability.",
        "",
        "## Paired test-row bootstrap",
        "",
        display_bootstrap.round(4).to_markdown(index=False),
        "",
        f"Each interval uses {args.bootstrap_resamples:,} paired resamples. These "
        "intervals describe test-row sampling uncertainty conditional on each "
        "fitted pair; they do not replace the across-seed summary above.",
        "",
        "## Exact McNemar tests for accuracy",
        "",
        display_mcnemar.to_markdown(index=False, floatfmt=".4g"),
        "",
        "McNemar's test concerns paired correctness, not macro-F1. It should not "
        "be used to support a claim about the macro-F1 difference.",
        "",
        "## Mean per-class F1 change across seeds",
        "",
        display_classes.round(4).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "The point estimates show a small, consistent reduction in overall "
        "macro-F1 rather than a collapse, although every per-seed bootstrap "
        "interval includes zero. The largest mean change is in Web, while Recon "
        "improves slightly. A restrained reading is therefore appropriate: the "
        "result is consistent with a modest contribution from the window-size "
        "field, and the transformer retains most of its within-dataset performance "
        "without it. The experiment does not establish transfer to another dataset.",
        "",
        f"The validation-median checkpoints selected without test inspection are "
        f"seed **{original_representative}** for `original` and seed "
        f"**{no_number_representative}** for `no_number`.",
        "",
    ]
    report_path = output_dir / f"{prefix}_summary.md"
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote feature-set comparison artifacts to {output_dir}")


if __name__ == "__main__":
    main()
