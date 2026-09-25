"""XGBoost/TreeSHAP arm of the explanation-robustness analysis (RQ4).

Section 4.4 perturbs 480 correctly classified test flows and re-attributes the
DistilBERT encoder with Layer Integrated Gradients (scripts/28). This script
applies the SAME deterministic perturbation to the SAME cohort and scores the
XGBoost baseline with exact TreeSHAP, so the two model families can be compared
under one protocol:

- cohort: data/revision/xai/main/robustness/robustness_cohort.parquet (480 rows,
  60 per class, drawn from rows DistilBERT classifies correctly);
- perturbation: copied verbatim from scripts/28_robustness_revised.py
  (`stable_rng`, `perturb`): x' = x * (1 + eps * N(0,1)) with a per-sample RNG
  derived from sha256(f"{seed}|{SampleID}|{eps:.12g}"); `proto_num` held fixed;
  the 22 fraction fields clipped to [0, 1]; the five count fields
  (ack_cnt, syn_cnt, fin_cnt, rst_cnt, num) rounded to integers and clipped at
  0; all other fields clipped at 0. eps in {1%, 5%, 10%}; seed 42;
- XGBoost receives the perturbed numeric vector directly (the encoder receives
  its %.4g serialization);
- attribution: shap.TreeExplainer toward the TRUE class, |values| L1-normalized
  per sample, exactly as scripts/28 does for Layer IG; metrics are Spearman over
  the 39 normalized |attribution| values and top-5 overlap/Jaccard.

Prediction preservation is reported over the rows XGBoost itself classifies
correctly before perturbation (the analogue of the encoder arm) and, for
completeness, as unchanged prediction over all 480 rows. The script also
records how often rounding leaves `num` unchanged at each eps.

Torch-free (xgboost + shap); seeded and deterministic; runs in about a minute.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from pipeline_config import CLASSES, feature_pairs


ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "data" / "revision"

# --- copied verbatim from scripts/28_robustness_revised.py -------------------
FRACTION_FIELDS = {
    "fin", "syn", "rst", "psh", "ack", "ece", "cwr", "http", "https",
    "dns", "telnet", "smtp", "ssh", "irc", "tcp", "udp", "dhcp", "arp",
    "icmp", "igmp", "ipv", "llc",
}
COUNT_FIELDS = {"ack_cnt", "syn_cnt", "fin_cnt", "rst_cnt", "num"}


def stable_rng(seed: int, sample_id: str, epsilon: float):
    material = f"{seed}|{sample_id}|{epsilon:.12g}".encode()
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(derived)


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


def exact_permutation_p(x, y):
    centered_x = np.asarray(x, dtype=np.float64) - np.mean(x)
    centered_y = np.asarray(y, dtype=np.float64) - np.mean(y)
    denominator = np.linalg.norm(centered_x) * np.linalg.norm(centered_y)
    if denominator == 0:
        return float("nan"), float("nan")
    observed = abs(float(centered_x @ centered_y / denominator))
    permutations = np.asarray(list(itertools.permutations(centered_y)))
    statistics = np.abs(permutations @ centered_x / denominator)
    return observed, float(np.mean(statistics >= observed - 1e-12))
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=REV / "runs" / "baselines" / "original" / "models" / "xgb.joblib")
    parser.add_argument("--test-split", type=Path, default=REV / "splits" / "test.parquet")
    parser.add_argument("--cohort", type=Path, default=REV / "xai" / "main" / "robustness" / "robustness_cohort.parquet")
    parser.add_argument("--encoder-records", type=Path, default=REV / "xai" / "main" / "robustness" / "robustness_records.parquet")
    parser.add_argument("--eps", type=float, nargs="+", default=(0.01, 0.05, 0.10))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=REV / "xai" / "main" / "robustness_xgboost")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_abs(values):
    absolute = np.abs(values)
    return absolute / (absolute.sum() or 1.0)


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "robustness_xgb_records.parquet").exists():
        raise FileExistsError("output already exists; choose a new --output-dir")

    pairs = feature_pairs("original")
    columns = [c for c, _ in pairs]
    names = [n for _, n in pairs]
    test = pd.read_parquet(args.test_split).reset_index(names="row")
    cohort = pd.read_parquet(args.cohort).sort_values("cohort_order")
    selected = cohort.merge(test, on=["row", "SampleID", "Category"], validate="one_to_one").sort_values("cohort_order")
    if len(selected) != len(cohort):
        raise ValueError("cohort did not align with the test split")

    model = joblib.load(args.model)
    explainer = shap.TreeExplainer(model)
    num_index = names.index("num")

    def attribute(matrix, targets):
        raw = np.asarray(explainer.shap_values(matrix.astype(np.float32)))
        if raw.shape == (len(matrix), len(names), len(CLASSES)):
            per_class = raw
        elif raw.shape == (len(matrix), len(CLASSES), len(names)):
            per_class = np.transpose(raw, (0, 2, 1))
        else:
            raise ValueError(f"unexpected SHAP shape {raw.shape}")
        return np.stack([normalized_abs(per_class[i, :, t]) for i, t in enumerate(targets)])

    base = selected[columns].to_numpy(np.float64)
    targets = selected["y_true"].to_numpy()
    base_pred = model.predict(base.astype(np.float32))
    base_attr = attribute(base, targets)

    records = []
    for epsilon in args.eps:
        changed = np.stack([
            perturb(base[i], names, epsilon, stable_rng(args.seed, sid, epsilon))
            for i, sid in enumerate(selected["SampleID"])
        ])
        pred = model.predict(changed.astype(np.float32))
        attr = attribute(changed, targets)
        for i, row in enumerate(selected.itertuples(index=False)):
            top_a = set(np.argsort(base_attr[i])[::-1][:5])
            top_b = set(np.argsort(attr[i])[::-1][:5])
            inter = len(top_a & top_b)
            rho = spearmanr(base_attr[i], attr[i]).correlation
            records.append({
                "cohort_order": int(row.cohort_order), "row": int(row.row), "SampleID": row.SampleID,
                "class": row.Category, "eps": float(epsilon),
                "xgb_base_correct": bool(base_pred[i] == targets[i]),
                "pred_unchanged": bool(pred[i] == base_pred[i]),
                "pred_preserved": bool(pred[i] == targets[i]),
                "num_unchanged": bool(changed[i, num_index] == base[i, num_index]),
                "spearman": float(rho) if np.isfinite(rho) else np.nan,
                "top5_overlap_count": inter,
                "top5_jaccard": inter / (10 - inter),
            })
    result = pd.DataFrame(records).sort_values(["cohort_order", "eps"])
    records_path = out / "robustness_xgb_records.parquet"
    result.to_parquet(records_path, index=False)

    correct = result[result["xgb_base_correct"]]
    metric_rows = []
    for class_name in CLASSES + ["_ALL"]:
        for epsilon in list(args.eps) + ["pooled"]:
            sub = correct if class_name == "_ALL" else correct[correct["class"] == class_name]
            allrows = result if class_name == "_ALL" else result[result["class"] == class_name]
            if epsilon != "pooled":
                sub = sub[sub["eps"] == epsilon]
                allrows = allrows[allrows["eps"] == epsilon]
            metric_rows.append({
                "class": class_name, "eps": epsilon, "n_base_correct": len(sub),
                "pred_preserved_frac": sub["pred_preserved"].mean(),
                "pred_unchanged_frac_all480": allrows["pred_unchanged"].mean(),
                "mean_spearman": sub["spearman"].mean(),
                "mean_top5_overlap": sub["top5_overlap_count"].mean(),
                "mean_top5_jaccard": sub["top5_jaccard"].mean(),
                "num_unchanged_frac": allrows["num_unchanged"].mean(),
            })
    metrics = pd.DataFrame(metric_rows)
    metrics_path = out / "robustness_xgb_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    per_class = metrics[(metrics["class"] != "_ALL") & (metrics["eps"] == "pooled")].set_index("class").loc[CLASSES]
    r_x, p_x = exact_permutation_p(per_class["pred_preserved_frac"], per_class["mean_top5_jaccard"])
    r_signed = float(np.corrcoef(per_class["pred_preserved_frac"], per_class["mean_top5_jaccard"])[0, 1])

    enc = pd.read_parquet(args.encoder_records)
    enc_overall = enc.groupby("eps")[["pred_preserved", "spearman", "top5_overlap_count", "top5_jaccard"]].mean()
    enc_class = enc.groupby("class")[["pred_preserved", "top5_jaccard", "spearman"]].mean().loc[CLASSES]

    lines = [
        "# Explanation robustness, XGBoost/TreeSHAP arm",
        "",
        f"Same 480-row cohort and the same deterministic perturbation as scripts/28 (seed {args.seed}; "
        "count fields rounded; proto_num fixed; fractions clipped to [0, 1]). TreeSHAP toward the true class, "
        "|values| L1-normalized per sample.",
        "",
        f"XGBoost classifies {int((base_pred == targets).sum())} of {len(targets)} cohort rows correctly before "
        "perturbation; prediction preservation and attribution metrics below are over those rows.",
        "",
        "## Overall",
        "",
        "| Perturbation | n | XGB prediction preserved | Mean Spearman | Mean top-5 overlap | DistilBERT preserved (same cohort) | DistilBERT top-5 overlap | `num` unchanged after rounding |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics[(metrics["class"] == "_ALL") & (metrics["eps"] != "pooled")].to_dict("records"):
        e = float(row["eps"])
        lines.append(
            f"| {e:.0%} | {row['n_base_correct']} | {row['pred_preserved_frac']:.1%} | {row['mean_spearman']:.3f} | "
            f"{row['mean_top5_overlap']:.2f}/5 | {enc_overall.loc[e, 'pred_preserved']:.1%} | "
            f"{enc_overall.loc[e, 'top5_overlap_count']:.2f}/5 | {row['num_unchanged_frac']:.1%} |"
        )
    lines += [
        "",
        "## By class, pooled across perturbation levels",
        "",
        "| Class | n rows | XGB preserved | XGB top-5 Jaccard | XGB Spearman | DistilBERT preserved | DistilBERT top-5 Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for class_name, row in per_class.iterrows():
        lines.append(
            f"| {class_name} | {int(row['n_base_correct']) // len(args.eps)} | {row['pred_preserved_frac']:.1%} | "
            f"{row['mean_top5_jaccard']:.3f} | {row['mean_spearman']:.3f} | "
            f"{enc_class.loc[class_name, 'pred_preserved']:.1%} | {enc_class.loc[class_name, 'top5_jaccard']:.3f} |"
        )
    lines += [
        "",
        f"XGBoost class-level Pearson r (prediction preserved vs top-5 Jaccard) = {r_signed:+.3f} "
        f"(exact permutation p = {p_x:.4f}; 8 classes, descriptive).",
        "",
        "The perturbation is benign multiplicative jitter, not an adversarial attack.",
        "",
    ]
    summary_path = out / "robustness_xgb_summary.md"
    summary_path.write_text("\n".join(lines))
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/39_robustness_xgboost.py",
        "seed": args.seed, "eps": list(args.eps),
        "xgb_base_correct": int((base_pred == targets).sum()),
        "class_level_pearson_r": r_signed, "class_level_permutation_p": p_x,
        "inputs": {k: {"path": str(Path(p).resolve().relative_to(ROOT)), "sha256": sha256(Path(p))}
                   for k, p in (("model", args.model), ("test_split", args.test_split),
                                ("cohort", args.cohort), ("encoder_records", args.encoder_records))},
        "outputs": {p.name: sha256(p) for p in (records_path, metrics_path, summary_path)},
    }
    (out / "robustness_xgb_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
