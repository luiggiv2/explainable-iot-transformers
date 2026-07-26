"""Phase 2 — classical baselines (RF, SVM, XGBoost) on tabular features.

8-class task (Category) over the 39 numeric features of the stratified subset.
Light hyperparameter selection on the validation split (documented below);
all final metrics reported on the untouched test split. Never accuracy alone:
per-class F1, macro/weighted aggregates, ROC-AUC (macro OvR) and the full
confusion matrix are produced for every model, plus train time, per-sample
inference latency and serialized model size (feeds RQ5).

Outputs (all under data/):
  baselines_metrics.csv       long table: model x class + aggregate rows
  confusion_{model}.csv       test confusion matrices (rows=true, cols=pred)
  baselines_summary.md        human-readable summary
  models/{model}.joblib       fitted models (reused for SHAP in Phase 4)
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = DATA / "models"

FEATURES = None  # resolved from columns (everything numeric except labels)
CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]


def load(split):
    df = pd.read_parquet(DATA / "splits" / f"{split}.parquet")
    X = df.drop(columns=["Label", "Category"]).to_numpy(dtype=np.float32)
    y = pd.Categorical(df["Category"], categories=CLASSES).codes
    return X, y


def evaluate(name, model, X_test, y_test, train_time):
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_ms = (time.perf_counter() - t0) / len(X_test) * 1e3

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)
    else:  # SVC without probability estimates
        scores = model.decision_function(X_test)
    auc = roc_auc_score(y_test, scores, multi_class="ovr", average="macro")

    rep = classification_report(y_test, y_pred, target_names=CLASSES,
                                output_dict=True, zero_division=0)
    rows = []
    for cls in CLASSES:
        r = rep[cls]
        rows.append({"model": name, "class": cls, "precision": r["precision"],
                     "recall": r["recall"], "f1": r["f1-score"],
                     "support": int(r["support"])})
    rows.append({"model": name, "class": "_accuracy", "f1": rep["accuracy"]})
    for agg in ("macro avg", "weighted avg"):
        r = rep[agg]
        rows.append({"model": name, "class": f"_{agg.replace(' ', '_')}",
                     "precision": r["precision"], "recall": r["recall"],
                     "f1": r["f1-score"]})
    rows.append({"model": name, "class": "_roc_auc_macro_ovr", "f1": auc})
    rows.append({"model": name, "class": "_train_time_s", "f1": train_time})
    rows.append({"model": name, "class": "_infer_ms_per_sample", "f1": infer_ms})

    path = MODELS / f"{name}.joblib"
    joblib.dump(model, path, compress=3)
    rows.append({"model": name, "class": "_model_size_mb",
                 "f1": path.stat().st_size / 1e6})

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        DATA / f"confusion_{name}.csv")
    print(f"{name}: macroF1={rep['macro avg']['f1-score']:.4f} "
          f"acc={rep['accuracy']:.4f} auc={auc:.4f} "
          f"train={train_time:.1f}s infer={infer_ms:.4f}ms/sample")
    return rows


def main():
    MODELS.mkdir(exist_ok=True)
    X_train, y_train = load("train")
    X_val, y_val = load("val")
    X_test, y_test = load("test")
    print(f"train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    all_rows = []

    # --- Random Forest (300 trees; depth unconstrained, standard for tabular IDS)
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    all_rows += evaluate("rf", rf, X_test, y_test, time.perf_counter() - t0)

    # --- XGBoost (hist, early stopping on val)
    t0 = time.perf_counter()
    xgb = XGBClassifier(
        n_estimators=1000, learning_rate=0.1, max_depth=8,
        tree_method="hist", eval_metric="mlogloss",
        early_stopping_rounds=30, random_state=SEED, n_jobs=-1,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"xgb best_iteration={xgb.best_iteration}")
    all_rows += evaluate("xgb", xgb, X_test, y_test, time.perf_counter() - t0)

    # --- SVM (RBF; C selected on val from {1, 10}; scaled inputs)
    best_f1, best_c = -1, None
    for C in (1.0, 10.0):
        t0 = time.perf_counter()
        svm = make_pipeline(StandardScaler(),
                            SVC(kernel="rbf", C=C, gamma="scale",
                                cache_size=1000, random_state=SEED))
        svm.fit(X_train, y_train)
        dt = time.perf_counter() - t0
        f1_val = f1_score(y_val, svm.predict(X_val), average="macro")
        print(f"svm C={C}: val macroF1={f1_val:.4f} ({dt:.0f}s)")
        if f1_val > best_f1:
            best_f1, best_c = f1_val, C
    # refit selected C with Platt-calibrated probabilities (needed for ROC-AUC)
    print(f"svm selected C={best_c}, refitting with probability=True")
    t0 = time.perf_counter()
    best = make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", C=best_c, gamma="scale",
                             cache_size=1000, probability=True,
                             random_state=SEED))
    best.fit(X_train, y_train)
    svm_time = time.perf_counter() - t0
    all_rows += evaluate("svm", best, X_test, y_test, svm_time)

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(DATA / "baselines_metrics.csv", index=False)

    # summary markdown
    per_class = metrics[~metrics["class"].str.startswith("_")]
    pivot = per_class.pivot(index="class", columns="model", values="f1").round(4)
    aggs = metrics[metrics["class"].str.startswith("_")]
    apivot = aggs.pivot(index="class", columns="model", values="f1").round(4)
    lines = [
        "# Classical baselines — CICIoT2023 8-class (test split, n=9,000)",
        "",
        f"Subset per data/subset_composition.md; seed {SEED}. "
        "RF: 300 trees. XGBoost: hist, lr 0.1, depth 8, early stopping on val. "
        f"SVM: RBF, C={best_c} selected on val, standardized inputs.",
        "",
        "## Per-class F1",
        pivot.to_markdown(),
        "",
        "## Aggregates / cost",
        apivot.to_markdown(),
        "",
        "Confusion matrices: confusion_rf.csv, confusion_xgb.csv, confusion_svm.csv",
        "",
    ]
    (DATA / "baselines_summary.md").write_text("\n".join(lines))
    print("wrote data/baselines_metrics.csv + baselines_summary.md")


if __name__ == "__main__":
    main()
