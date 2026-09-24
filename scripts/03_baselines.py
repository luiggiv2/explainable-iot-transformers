"""Train leakage-safe classical baselines and registered feature ablations.

The script accepts an explicit split directory, feature set, and run directory.
Every model writes aligned test predictions, probabilities, a confusion matrix,
and a checksum manifest beside its metrics. Inference timings are diagnostic;
RQ5 uses a separate controlled benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from pipeline_config import CLASSES, FEATURE_SET_EXCLUSIONS, feature_pairs


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DATA / "splits")
    parser.add_argument("--run-dir", type=Path, default=DATA)
    parser.add_argument("--feature-set", choices=FEATURE_SET_EXCLUSIONS, default="original")
    parser.add_argument(
        "--models",
        default="rf,xgb,svm",
        help="Comma-separated subset of rf,xgb,svm.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing model files already present in the run directory.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_split(path: Path, columns):
    frame = pd.read_parquet(path).reset_index(drop=True)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing features {sorted(missing)}")
    X = frame[columns].to_numpy(dtype=np.float32)
    y = pd.Categorical(frame["Category"], categories=CLASSES).codes
    if np.any(y < 0):
        raise ValueError(f"unknown category in {path}")
    return frame, X, np.asarray(y)


def metric_rows(name, report, auc, train_time, infer_ms, model_size_mb):
    rows = []
    for cls in CLASSES:
        values = report[cls]
        rows.append(
            {
                "model": name,
                "class": cls,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": int(values["support"]),
            }
        )
    rows.append({"model": name, "class": "_accuracy", "f1": report["accuracy"]})
    for aggregate in ("macro avg", "weighted avg"):
        values = report[aggregate]
        rows.append(
            {
                "model": name,
                "class": f"_{aggregate.replace(' ', '_')}",
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
            }
        )
    rows.extend(
        [
            {"model": name, "class": "_roc_auc_macro_ovr", "f1": auc},
            {"model": name, "class": "_train_time_s", "f1": train_time},
            {
                "model": name,
                "class": "_diagnostic_infer_ms_per_sample",
                "f1": infer_ms,
            },
            {"model": name, "class": "_model_size_mb", "f1": model_size_mb},
        ]
    )
    return rows


def evaluate(name, model, test_frame, X_test, y_test, train_time, run_dir, model_dir):
    started = time.perf_counter()
    y_pred = np.asarray(model.predict(X_test))
    infer_ms = (time.perf_counter() - started) / len(X_test) * 1e3
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{name} must expose predict_proba for comparable ROC-AUC")
    probabilities = np.asarray(model.predict_proba(X_test))
    auc = roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro")
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    model_path = model_dir / f"{name}.joblib"
    joblib.dump(model, model_path, compress=3)
    matrix = confusion_matrix(y_test, y_pred, labels=np.arange(len(CLASSES)))
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(
        run_dir / f"confusion_{name}.csv"
    )
    records = pd.DataFrame(
        {
            "row": np.arange(len(test_frame)),
            "subtype": test_frame["Label"].to_numpy(),
            "true_category": test_frame["Category"].to_numpy(),
            "predicted_category": np.asarray(CLASSES, dtype=object)[y_pred],
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    if "SampleID" in test_frame:
        records.insert(1, "SampleID", test_frame["SampleID"].to_numpy())
    for index, cls in enumerate(CLASSES):
        records[f"prob_{cls}"] = probabilities[:, index]
    prediction_path = run_dir / f"test_predictions_{name}.parquet"
    records.to_parquet(prediction_path, index=False)
    assert np.array_equal(
        matrix,
        confusion_matrix(
            records["y_true"], records["y_pred"], labels=np.arange(len(CLASSES))
        ),
    )

    size_mb = model_path.stat().st_size / 1e6
    rows = metric_rows(name, report, auc, train_time, infer_ms, size_mb)
    print(
        f"{name}: macroF1={report['macro avg']['f1-score']:.4f} "
        f"acc={report['accuracy']:.4f} auc={auc:.4f} "
        f"train={train_time:.1f}s"
    )
    return rows, {
        "model": sha256(model_path),
        "predictions": sha256(prediction_path),
        "confusion_reconstructed_equal": True,
    }


def main():
    args = parse_args()
    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = set(requested) - {"rf", "xgb", "svm"}
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    if not requested:
        raise ValueError("--models selected no models")

    split_dir = args.split_dir.resolve()
    run_dir = args.run_dir.resolve()
    model_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    existing = [model_dir / f"{name}.joblib" for name in requested]
    existing = [path for path in existing if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"model files already exist: {existing}; pass --overwrite explicitly"
        )

    columns = [column for column, _ in feature_pairs(args.feature_set)]
    train_frame, X_train, y_train = load_split(split_dir / "train.parquet", columns)
    val_frame, X_val, y_val = load_split(split_dir / "val.parquet", columns)
    test_frame, X_test, y_test = load_split(split_dir / "test.parquet", columns)
    print(
        f"feature_set {args.feature_set}: train {X_train.shape}, "
        f"val {X_val.shape}, test {X_test.shape}"
    )

    all_rows = []
    checksums = {}
    selected_svm_c = None
    if "rf" in requested:
        started = time.perf_counter()
        model = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=args.seed
        )
        model.fit(X_train, y_train)
        rows, model_hashes = evaluate(
            "rf",
            model,
            test_frame,
            X_test,
            y_test,
            time.perf_counter() - started,
            run_dir,
            model_dir,
        )
        all_rows.extend(rows)
        checksums["rf"] = model_hashes

    if "xgb" in requested:
        started = time.perf_counter()
        model = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=8,
            tree_method="hist",
            eval_metric="mlogloss",
            early_stopping_rounds=30,
            random_state=args.seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print(f"xgb best_iteration={model.best_iteration}")
        rows, model_hashes = evaluate(
            "xgb",
            model,
            test_frame,
            X_test,
            y_test,
            time.perf_counter() - started,
            run_dir,
            model_dir,
        )
        all_rows.extend(rows)
        checksums["xgb"] = model_hashes

    if "svm" in requested:
        best_f1, selected_svm_c = -1.0, None
        for candidate in (1.0, 10.0):
            model = make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="rbf",
                    C=candidate,
                    gamma="scale",
                    cache_size=1000,
                    random_state=args.seed,
                ),
            )
            model.fit(X_train, y_train)
            val_f1 = f1_score(y_val, model.predict(X_val), average="macro")
            print(f"svm C={candidate}: val macroF1={val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1, selected_svm_c = val_f1, candidate
        started = time.perf_counter()
        model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=selected_svm_c,
                gamma="scale",
                cache_size=1000,
                probability=True,
                random_state=args.seed,
            ),
        )
        model.fit(X_train, y_train)
        rows, model_hashes = evaluate(
            "svm",
            model,
            test_frame,
            X_test,
            y_test,
            time.perf_counter() - started,
            run_dir,
            model_dir,
        )
        all_rows.extend(rows)
        checksums["svm"] = model_hashes

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(run_dir / "baselines_metrics.csv", index=False)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "feature_set": args.feature_set,
        "features": columns,
        "seed": args.seed,
        "models": requested,
        "selected_svm_c": selected_svm_c,
        "split_dir": str(split_dir.relative_to(ROOT)),
        "rows": {
            "train": len(train_frame),
            "validation": len(val_frame),
            "test": len(test_frame),
        },
        "split_sha256": {
            name: sha256(split_dir / f"{name}.parquet")
            for name in ("train", "val", "test")
        },
        "artifacts": checksums,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    per_class = metrics[~metrics["class"].str.startswith("_")]
    aggregate = metrics[metrics["class"].str.startswith("_")]
    summary = [
        "# Classical baselines — CICIoT2023 8-class",
        "",
        f"Feature set `{args.feature_set}`; seed {args.seed}; leakage-safe grouped "
        f"split; models {requested}.",
        "",
        "## Per-class F1",
        per_class.pivot(index="class", columns="model", values="f1")
        .round(4)
        .to_markdown(),
        "",
        "## Aggregates and diagnostic cost",
        aggregate.pivot(index="class", columns="model", values="f1")
        .round(4)
        .to_markdown(),
        "",
        "Predictions and probabilities are stored per model and reconstruct every "
        "reported confusion matrix. Latencies are diagnostic only.",
        "",
    ]
    (run_dir / "baselines_summary.md").write_text("\n".join(summary))
    print(f"wrote complete baseline run to {run_dir}")


if __name__ == "__main__":
    main()
