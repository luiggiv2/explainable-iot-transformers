"""Seed variance of the tree baselines, compared with DistilBERT's three seeds.

The classical baselines in Section 4.1 were fitted once (seed 42). This script
summarizes re-fits of Random Forest and XGBoost produced by
`scripts/03_baselines.py --models rf,xgb --seed S --run-dir
data/revision/runs/baselines_seeds/seed_S` for S in {42, 123, 2026} (same fixed
configuration, same leakage-safe splits), checks that seed 42 reproduces the
reported run exactly, reads XGBoost's early-stopping iteration and validation
mlogloss from the saved models, and sets the across-seed spread beside that of
DistilBERT. Torch-free (joblib/xgboost only); no training happens here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "data" / "revision"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-dir", type=Path, default=REV / "runs" / "baselines_seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--original-run", type=Path, default=REV / "runs" / "baselines" / "original")
    parser.add_argument("--distilbert-seeds", type=Path, default=REV / "results" / "distilbert_original_seeds.csv")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metrics(run_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "baselines_metrics.csv")
    pick = metrics[metrics["class"].isin(["_macro_avg", "_accuracy", "_roc_auc_macro_ovr", "_weighted_avg"])]
    return pick.pivot(index="model", columns="class", values="f1")


def xgb_early_stopping(model_path: Path) -> dict:
    model = joblib.load(model_path)
    evals = model.evals_result()
    history = evals.get("validation_0", {}).get("mlogloss", [])
    return {
        "best_iteration": int(model.best_iteration),
        "best_score_mlogloss": float(model.best_score),
        "boosting_rounds_fitted": len(history),
    }


def main():
    args = parse_args()
    rows = []
    early = {}
    inputs = {}
    for seed in args.seeds:
        run = args.seeds_dir / f"seed_{seed}"
        table = read_metrics(run)
        inputs[f"seed_{seed}_metrics"] = run / "baselines_metrics.csv"
        for model in table.index:
            rows.append({
                "model": model, "seed": seed,
                "macro_f1": table.loc[model, "_macro_avg"],
                "accuracy": table.loc[model, "_accuracy"],
                "weighted_f1": table.loc[model, "_weighted_avg"],
                "roc_auc": table.loc[model, "_roc_auc_macro_ovr"],
            })
        early[f"seed_{seed}"] = xgb_early_stopping(run / "models" / "xgb.joblib")
    early["original_run"] = xgb_early_stopping(args.original_run / "models" / "xgb.joblib")
    per_seed = pd.DataFrame(rows)

    db = pd.read_csv(args.distilbert_seeds)
    inputs["distilbert_seeds"] = args.distilbert_seeds
    f1_col = next(c for c in db.columns if "macro_f1" in c and "test" in c)
    acc_col = next(c for c in db.columns if "accuracy" in c and "test" in c)
    for record in db.to_dict("records"):
        per_seed = pd.concat([per_seed, pd.DataFrame([{
            "model": "distilbert", "seed": int(record["seed"]),
            "macro_f1": record[f1_col], "accuracy": record[acc_col],
        }])], ignore_index=True)

    summary = (per_seed.groupby("model")
               .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                    macro_f1_min=("macro_f1", "min"), macro_f1_max=("macro_f1", "max"),
                    accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"))
               .reset_index())

    original = read_metrics(args.original_run)
    inputs["original_metrics"] = args.original_run / "baselines_metrics.csv"
    reproduces = {
        model: bool(np.isclose(original.loc[model, "_macro_avg"],
                               per_seed.query("model == @model and seed == 42")["macro_f1"].iloc[0], atol=0, rtol=0))
        for model in ("rf", "xgb")
    }

    out = args.seeds_dir
    per_seed.to_csv(out / "baseline_seed_metrics.csv", index=False)
    summary.to_csv(out / "baseline_seed_summary.csv", index=False)
    lines = [
        "# Seed variance: tree baselines vs DistilBERT",
        "",
        "Random Forest and XGBoost re-fitted with `scripts/03_baselines.py` (fixed configuration, "
        "seeds 42/123/2026, same splits). DistilBERT values are the three fine-tuning runs "
        "(`results/distilbert_original_seeds.csv`).",
        "",
        "| Model | Seed | Macro-F1 | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for row in per_seed.sort_values(["model", "seed"]).to_dict("records"):
        lines.append(f"| {row['model']} | {row['seed']} | {row['macro_f1']:.4f} | {row['accuracy']:.4f} |")
    lines += ["", "| Model | macro-F1 mean ± SD | range | accuracy mean ± SD |", "|---|---:|---|---:|"]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['model']} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_sd']:.4f} | "
                     f"[{row['macro_f1_min']:.4f}, {row['macro_f1_max']:.4f}] | "
                     f"{row['accuracy_mean']:.4f} ± {row['accuracy_sd']:.4f} |")
    lines += [
        "",
        f"Seed 42 reproduces the reported baseline run exactly: RF {reproduces['rf']}, XGBoost {reproduces['xgb']}.",
        "",
        "## XGBoost early stopping (validation mlogloss)",
        "",
        "| Run | best iteration (0-based) | best validation mlogloss | rounds fitted |",
        "|---|---:|---:|---:|",
    ]
    for key, value in early.items():
        lines.append(f"| {key} | {value['best_iteration']} | {value['best_score_mlogloss']:.4f} | {value['boosting_rounds_fitted']} |")
    lines += [
        "",
        "XGBoost uses no row or column subsampling in this configuration, so its seed has no effect on the fitted "
        "model when the outputs are identical across seeds.",
        "",
    ]
    summary_path = out / "baseline_seed_summary.md"
    summary_path.write_text("\n".join(lines))
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/38_baseline_seed_summary.py",
        "producer": "scripts/03_baselines.py --models rf,xgb --feature-set original --seed S",
        "seeds": args.seeds,
        "seed42_reproduces_reported": reproduces,
        "xgb_early_stopping": early,
        "inputs": {k: {"path": str(Path(p).resolve().relative_to(ROOT)), "sha256": sha256(Path(p))} for k, p in inputs.items()},
        "outputs": {p.name: sha256(p) for p in (out / "baseline_seed_metrics.csv", out / "baseline_seed_summary.csv", summary_path)},
    }
    (out / "baseline_seed_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
