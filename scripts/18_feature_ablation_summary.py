"""Summarize registered tree ablations and quantify simple shortcut signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier

from pipeline_config import CLASSES


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEATURE_SETS = ("original", "no_number", "no_protocol", "no_number_protocol")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DATA / "revision" / "runs" / "baselines",
    )
    parser.add_argument(
        "--split-dir", type=Path, default=DATA / "revision" / "splits"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DATA / "revision" / "results"
    )
    return parser.parse_args()


def load_metric_table(root: Path):
    frames = []
    for feature_set in FEATURE_SETS:
        path = root / feature_set / "baselines_metrics.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "feature_set", feature_set)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def shortcut_models(split_dir: Path):
    train = pd.read_parquet(split_dir / "train.parquet")
    test = pd.read_parquet(split_dir / "test.parquet")
    y_train = pd.Categorical(train["Category"], categories=CLASSES).codes
    y_test = pd.Categorical(test["Category"], categories=CLASSES).codes
    settings = {
        "class-prior dummy": [],
        "Number only": ["Number"],
        "Protocol Type only": ["Protocol Type"],
        "Number + Protocol Type": ["Number", "Protocol Type"],
    }
    rows = []
    per_class = []
    for name, columns in settings.items():
        if columns:
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
            model.fit(train[columns], y_train)
            pred = model.predict(test[columns])
        else:
            model = DummyClassifier(strategy="prior", random_state=42)
            model.fit(np.zeros((len(train), 1)), y_train)
            pred = model.predict(np.zeros((len(test), 1)))
        report = classification_report(
            y_test,
            pred,
            labels=np.arange(len(CLASSES)),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )
        rows.append(
            {
                "diagnostic": name,
                "features": ", ".join(columns) or "none",
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
            }
        )
        for cls in CLASSES:
            per_class.append(
                {
                    "diagnostic": name,
                    "class": cls,
                    "f1": report[cls]["f1-score"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(per_class), train, test


def main():
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metric_table(baseline_root)
    macro = metrics[metrics["class"] == "_macro_avg"][
        ["feature_set", "model", "f1"]
    ].rename(columns={"f1": "macro_f1"})
    original = macro[macro["feature_set"] == "original"].set_index("model")[
        "macro_f1"
    ]
    macro["delta_vs_original"] = macro.apply(
        lambda row: row["macro_f1"] - original[row["model"]], axis=1
    )
    macro.to_csv(output_dir / "feature_ablation_summary.csv", index=False)

    per_class = metrics[~metrics["class"].str.startswith("_")][
        ["feature_set", "model", "class", "f1"]
    ].copy()
    original_pc = per_class[per_class["feature_set"] == "original"].set_index(
        ["model", "class"]
    )["f1"]
    per_class["delta_vs_original"] = per_class.apply(
        lambda row: row["f1"] - original_pc.loc[(row["model"], row["class"])],
        axis=1,
    )
    per_class.to_csv(output_dir / "feature_ablation_per_class.csv", index=False)

    diagnostics, diagnostic_pc, train, test = shortcut_models(split_dir)
    diagnostics.to_csv(output_dir / "shortcut_diagnostics.csv", index=False)
    diagnostic_pc.to_csv(output_dir / "shortcut_diagnostics_per_class.csv", index=False)

    number_distribution = (
        test.groupby("Category")["Number"]
        .agg(["count", "mean", "min", "max", "nunique"])
        .reindex(CLASSES)
    )
    regime = pd.crosstab(
        test["Category"],
        np.where(test["Number"] < 50, "short (<50)", "long (>=50)"),
        normalize="index",
    ).reindex(CLASSES)

    largest_class_changes = (
        per_class[per_class["feature_set"] != "original"]
        .assign(abs_delta=lambda frame: frame["delta_vs_original"].abs())
        .sort_values("abs_delta", ascending=False)
        .head(12)
    )
    threshold_crossed = bool(
        (macro[macro["feature_set"] != "original"]["delta_vs_original"].abs() >= 0.01).any()
    )
    lines = [
        "# Registered feature ablations and shortcut diagnostics",
        "",
        "All results use the leakage-safe grouped split. Tree ablations change only "
        "the registered input columns; hyperparameters and test rows remain fixed.",
        "",
        "## Macro-F1",
        "",
        macro.pivot(index="feature_set", columns="model", values="macro_f1")
        .round(4)
        .to_markdown(),
        "",
        "## Macro-F1 delta from the original feature set",
        "",
        macro.pivot(index="feature_set", columns="model", values="delta_vs_original")
        .round(4)
        .to_markdown(),
        "",
        f"The pre-registered expansion gate (absolute macro-F1 change >= 0.01) was "
        f"{'triggered' if threshold_crossed else 'not triggered'} for the tree models.",
        "",
        "## Simple shortcut diagnostics",
        "",
        diagnostics.round(4).to_markdown(index=False),
        "",
        "These depth-3 trees measure how much label information is available from the "
        "suspect variables alone; they are diagnostics, not competitive baselines.",
        "",
        "## `Number` distribution by category (test split)",
        "",
        number_distribution.round(3).to_markdown(),
        "",
        "## Short/long-window regime fraction by category",
        "",
        regime.fillna(0).round(3).to_markdown(),
        "",
        "## Largest per-class ablation changes",
        "",
        largest_class_changes[
            ["feature_set", "model", "class", "f1", "delta_vs_original"]
        ]
        .round(4)
        .to_markdown(index=False),
        "",
    ]
    (output_dir / "feature_ablation_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote ablation diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
