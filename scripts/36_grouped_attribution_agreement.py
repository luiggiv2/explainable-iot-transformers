"""Feature redundancy among the 39 serialized fields and group-level IG/SHAP agreement.

The field-level comparison of Layer Integrated Gradients (DistilBERT) and
TreeSHAP (XGBoost) in Section 4.2 treats the 39 fields as distinct evidence.
Several of them are, however, algebraically linked or near-monotone copies of
one another in CICIoT2023 (for example `Tot size` and `AVG`). Two models can
then split the same information across different members of a redundant set,
which a field-level rank comparison scores as disagreement.

This script, on the TRAIN split only:

1. computes the 39x39 Spearman correlation matrix of the raw feature columns;
2. checks exact algebraic identities (Tot size == AVG, Tot sum == Number*AVG,
   Variance == Std^2, IPv == LLC == 1 - ARP) and the linear dependence of
   `Protocol Type` on the protocol indicators;
3. forms redundancy groups as connected components of the graph whose edges are
   |rho| > threshold (default 0.9) or an exact pairwise identity (grouping G1),
   and a second grouping G2 that additionally merges the three-way identity
   Tot sum = Number * AVG;
4. recomputes the per-class IG-vs-SHAP agreement (Spearman over units and top-5
   overlap) at field level (reproducing the manuscript numbers), at G1 and at
   G2 level.

Aggregation (primary): a group's class-level attribution is the SUM of its
members' class-mean absolute L1-normalized attributions, read from the saved
per-class tables. Sensitivity: the class mean over samples of |sum of the
members' signed per-sample attributions| (allows within-group cancellation),
read from the per-sample tables. No model inference is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline_config import CLASSES, FEATURE_PAIRS


ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "data" / "revision"
NAME = dict(FEATURE_PAIRS)
FIELDS = [name for _, name in FEATURE_PAIRS]
INDICATORS = ["HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
              "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=REV / "splits" / "train.parquet")
    parser.add_argument("--ig-table", type=Path,
                        default=REV / "xai" / "main" / "distilbert" / "captum_feature_attribution.csv")
    parser.add_argument("--shap-table", type=Path,
                        default=REV / "xai" / "main" / "xgboost" / "shap_feature_attribution.csv")
    parser.add_argument("--ig-per-sample", type=Path,
                        default=REV / "xai" / "main" / "distilbert" / "captum_per_sample.parquet")
    parser.add_argument("--shap-per-sample", type=Path,
                        default=REV / "xai" / "main" / "xgboost" / "shap_per_sample.parquet")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path,
                        default=REV / "xai" / "main" / "grouped_agreement")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def components(nodes, edges):
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in edges:
        parent[find(a)] = find(b)
    groups = {}
    for node in nodes:
        groups.setdefault(find(node), []).append(node)
    # keep serialization order inside and across groups
    ordered = sorted(groups.values(), key=lambda members: FIELDS.index(members[0]))
    return [sorted(members, key=FIELDS.index) for members in ordered]


def top_overlap(a: pd.Series, b: pd.Series, k: int) -> int:
    return len(set(a.nlargest(k).index) & set(b.nlargest(k).index))


def agreement(ig: pd.DataFrame, shap: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    for class_name in CLASSES:
        a, b = ig.loc[class_name], shap.loc[class_name]
        rows.append({
            "class": class_name,
            "units": len(a),
            "spearman": float(spearmanr(a.to_numpy(), b.to_numpy()).correlation),
            "top_overlap": top_overlap(a, b, k),
            "ig_top": a.idxmax(),
            "shap_top": b.idxmax(),
            "shap_exact_zero_units": int((b == 0).sum()),
        })
    return pd.DataFrame(rows)


def group_label(members):
    return "+".join(members)


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.train)
    columns = [column for column, _ in FEATURE_PAIRS]
    raw = train[columns].astype(np.float64)
    rho = pd.DataFrame(spearmanr(raw.to_numpy()).correlation, index=FIELDS, columns=FIELDS)
    rho = rho.fillna(0.0)
    rho.to_csv(out / "train_spearman_matrix.csv")

    # ---- algebraic identities (train) -------------------------------------
    def rel_err(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        scale = np.maximum(np.abs(b), 1e-12)
        return float(np.max(np.abs(a - b) / scale))

    identities = []
    identities.append({"identity": "Tot size == AVG",
                       "fraction_exact": float((train["Tot size"] == train["AVG"]).mean()),
                       "max_relative_error": rel_err(train["Tot size"], train["AVG"])})
    nz = train["AVG"] != 0
    identities.append({"identity": "Tot sum == Number * AVG",
                       "fraction_exact": float(np.isclose(train["Tot sum"], train["Number"] * train["AVG"], rtol=1e-9, atol=1e-9).mean()),
                       "max_relative_error": rel_err(train["Tot sum"], train["Number"] * train["AVG"])})
    identities.append({"identity": "Tot sum / AVG == Number (rows with AVG != 0)",
                       "fraction_exact": float(np.isclose(train.loc[nz, "Tot sum"] / train.loc[nz, "AVG"], train.loc[nz, "Number"], rtol=1e-9).mean()),
                       "max_relative_error": rel_err(train.loc[nz, "Tot sum"] / train.loc[nz, "AVG"], train.loc[nz, "Number"])})
    identities.append({"identity": "Variance == Std^2",
                       "fraction_exact": float(np.isclose(train["Variance"], train["Std"] ** 2, rtol=1e-6, atol=1e-6).mean()),
                       "max_relative_error": rel_err(train["Variance"], train["Std"] ** 2)})
    identities.append({"identity": "IPv == LLC",
                       "fraction_exact": float((train["IPv"] == train["LLC"]).mean()),
                       "max_relative_error": rel_err(train["IPv"], train["LLC"])})
    identities.append({"identity": "IPv == 1 - ARP",
                       "fraction_exact": float(np.isclose(train["IPv"], 1 - train["ARP"], atol=1e-9).mean()),
                       "max_relative_error": float(np.max(np.abs(train["IPv"] - (1 - train["ARP"]))))})
    ident = pd.DataFrame(identities)
    ident.to_csv(out / "train_identities.csv", index=False)

    x = np.c_[train[INDICATORS].to_numpy(float), np.ones(len(train))]
    y = train["Protocol Type"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    r2_proto = float(1 - np.sum((y - x @ coef) ** 2) / np.sum((y - y.mean()) ** 2))
    proto_rho = rho.loc["proto_num", [NAME[c] for c in INDICATORS]].sort_values(key=np.abs, ascending=False)

    # ---- grouping -----------------------------------------------------------
    corr_edges = [(a, b) for a, b in combinations(FIELDS, 2) if abs(rho.loc[a, b]) > args.threshold]
    pairwise_identity_edges = [("tot_size", "avg"), ("var", "std"), ("ipv", "llc"), ("arp", "ipv")]
    g1 = components(FIELDS, corr_edges + pairwise_identity_edges)
    g2 = components(FIELDS, corr_edges + pairwise_identity_edges + [("tot_sum", "num"), ("tot_sum", "avg")])
    edges = pd.DataFrame(
        [{"a": a, "b": b, "spearman": float(rho.loc[a, b])} for a, b in corr_edges]
    )
    edges.to_csv(out / "train_high_correlation_pairs.csv", index=False)

    # ---- attribution tables ---------------------------------------------------
    ig_tab = pd.read_csv(args.ig_table).pivot(index="class", columns="feature", values="abs_mean")[FIELDS]
    shap_tab = pd.read_csv(args.shap_table).pivot(index="class", columns="feature", values="abs_mean")[FIELDS]
    ig_ps = pd.read_parquet(args.ig_per_sample)
    shap_ps = pd.read_parquet(args.shap_per_sample)

    def grouped_sum(table, groups):
        return pd.DataFrame({group_label(g): table[g].sum(axis=1) for g in groups})

    def grouped_signed(per_sample, groups):
        frame = pd.DataFrame({group_label(g): per_sample[g].sum(axis=1).abs() for g in groups})
        frame["class"] = per_sample["class"].to_numpy()
        return frame.groupby("class").mean().loc[CLASSES]

    results = {}
    results["field"] = agreement(ig_tab.loc[CLASSES], shap_tab.loc[CLASSES], args.top_k)
    for label, groups in (("G1", g1), ("G2", g2)):
        results[f"{label}_sum_abs"] = agreement(grouped_sum(ig_tab, groups).loc[CLASSES],
                                                grouped_sum(shap_tab, groups).loc[CLASSES], args.top_k)
        results[f"{label}_abs_signed_sum"] = agreement(grouped_signed(ig_ps, groups),
                                                       grouped_signed(shap_ps, groups), args.top_k)
    frames = []
    for level, frame in results.items():
        frame = frame.copy()
        frame.insert(0, "level", level)
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(out / "grouped_agreement_per_class.csv", index=False)

    summary = (
        long.groupby("level", sort=False)
        .agg(units=("units", "first"), mean_spearman=("spearman", "mean"),
             min_spearman=("spearman", "min"), max_spearman=("spearman", "max"),
             mean_top_overlap=("top_overlap", "mean"))
        .reset_index()
    )
    summary["mean_top_overlap_pct"] = summary["mean_top_overlap"] / args.top_k
    summary["chance_top_overlap_pct"] = args.top_k / summary["units"]
    summary.to_csv(out / "grouped_agreement_summary.csv", index=False)

    # group ranks of selected units per class under G1 (sum of |attr|)
    rank_rows = []
    for label, groups in (("G1", g1), ("G2", g2)):
        ig_g = grouped_sum(ig_tab, groups).loc[CLASSES]
        sh_g = grouped_sum(shap_tab, groups).loc[CLASSES]
        for class_name in CLASSES:
            ig_rank = ig_g.loc[class_name].rank(ascending=False, method="min")
            sh_rank = sh_g.loc[class_name].rank(ascending=False, method="min")
            for unit in ig_g.columns:
                rank_rows.append({
                    "grouping": label, "class": class_name, "unit": unit,
                    "ig_abs": float(ig_g.loc[class_name, unit]), "ig_rank": int(ig_rank[unit]),
                    "shap_abs": float(sh_g.loc[class_name, unit]), "shap_rank": int(sh_rank[unit]),
                })
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(out / "grouped_ranks_per_class.csv", index=False)

    # ---- report -----------------------------------------------------------------
    lines = [
        "# Feature redundancy and group-level IG/SHAP agreement",
        "",
        f"Train split: {len(train):,} rows, 39 fields. Spearman correlations and identities are computed on TRAIN only.",
        "",
        "## Exact algebraic identities (train)",
        "",
        "| Identity | fraction of rows satisfied | max error |",
        "|---|---:|---:|",
    ]
    for row in ident.itertuples(index=False):
        lines.append(f"| {row.identity} | {row.fraction_exact:.4f} | {row.max_relative_error:.2e} |")
    lines += [
        "",
        f"`Protocol Type` regressed linearly on the 15 protocol indicators: R^2 = {r2_proto:.4f}. "
        "Largest |Spearman| with an indicator: "
        + ", ".join(f"`{k}` {v:+.3f}" for k, v in proto_rho.head(4).items()) + ".",
        "",
        f"## Field pairs with |Spearman| > {args.threshold} (train)",
        "",
        "| a | b | Spearman |",
        "|---|---|---:|",
    ]
    for row in edges.itertuples(index=False):
        lines.append(f"| `{row.a}` | `{row.b}` | {row.spearman:+.3f} |")

    def render_groups(groups):
        multi = [g for g in groups if len(g) > 1]
        return ", ".join("{" + ", ".join(f"`{m}`" for m in g) + "}" for g in multi)

    lines += [
        "",
        "## Groupings",
        "",
        f"- **G1** (|rho| > {args.threshold} or exact pairwise identity; {len(g1)} units): "
        f"multi-field groups {render_groups(g1)}; all other fields are singletons.",
        f"- **G2** (G1 plus the three-way identity Tot sum = Number x AVG; {len(g2)} units): "
        f"multi-field groups {render_groups(g2)}.",
        "",
        "## IG vs SHAP agreement per level",
        "",
        "`sum_abs`: group score = sum of members' class-mean |attribution| (primary). "
        "`abs_signed_sum`: class mean over samples of |sum of members' signed attribution| (sensitivity). "
        f"Top-{args.top_k} overlap chance level = {args.top_k}/units.",
        "",
        "| Level | units | mean Spearman | range | mean top-5 overlap | chance |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.level} | {row.units} | {row.mean_spearman:+.3f} | "
            f"[{row.min_spearman:+.3f}, {row.max_spearman:+.3f}] | "
            f"{row.mean_top_overlap:.2f}/5 ({row.mean_top_overlap_pct:.1%}) | {row.chance_top_overlap_pct:.1%} |"
        )
    lines += ["", "## Per class", "", "| Level | Class | Spearman | top-5 overlap | IG top | SHAP top | SHAP exact-zero units |",
              "|---|---|---:|---:|---|---|---:|"]
    for row in long.to_dict("records"):
        lines.append(f"| {row['level']} | {row['class']} | {row['spearman']:+.3f} | "
                     f"{row['top_overlap']}/5 | `{row['ig_top']}` | `{row['shap_top']}` | {row['shap_exact_zero_units']} |")
    focus = ranks[(ranks["grouping"] == "G1") & ranks["unit"].str.contains("arp|num|tot_size|tot_sum|icmp|ssh|proto_num")]
    lines += ["", "## Selected G1 unit ranks (IG rank / SHAP rank)", "",
              "| Class | Unit | IG rank | IG |attr| sum | SHAP rank | SHAP |attr| sum |".replace("|attr|", "abs"),
              "|---|---|---:|---:|---:|---:|"]
    for row in focus.to_dict("records"):
        lines.append(f"| {row['class']} | `{row['unit']}` | {row['ig_rank']} | {row['ig_abs']:.4f} | {row['shap_rank']} | {row['shap_abs']:.4f} |")
    lines.append("")
    summary_path = out / "grouped_agreement.md"
    summary_path.write_text("\n".join(lines))

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/36_grouped_attribution_agreement.py",
        "threshold": args.threshold,
        "top_k": args.top_k,
        "groupings": {"G1": g1, "G2": g2},
        "protocol_type_linear_r2": r2_proto,
        "inputs": {
            key: {"path": str(path.resolve().relative_to(ROOT)), "sha256": sha256(path)}
            for key, path in (("train", args.train), ("ig_table", args.ig_table),
                              ("shap_table", args.shap_table), ("ig_per_sample", args.ig_per_sample),
                              ("shap_per_sample", args.shap_per_sample))
        },
        "outputs": {path.name: sha256(path) for path in sorted(out.glob("*")) if path.suffix in {".csv", ".md"}},
    }
    (out / "grouped_agreement_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
