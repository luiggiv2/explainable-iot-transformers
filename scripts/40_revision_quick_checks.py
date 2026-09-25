"""Small consistency checks requested during manuscript revision.

Each check reads existing artifacts (or the leakage-safe splits) and persists
the number the manuscript quotes, so every such number traces to a file:

1. BruteForce windows with any SSH traffic (SSH > 0), train/test and XAI cohort;
2. Recon subtype composition on the test split;
3. batch-1 -> batch-64 speed-up per model from the controlled CPU benchmark;
4. Layer-IG convergence-delta statistics from the saved per-sample file;
5. McNemar p-values of the no_number vs original comparison, per seed;
6. share of the summed per-class F1 loss from removing `Number` that falls on
   Web and BruteForce;
7. exact-feature duplicate groups whose rows carry conflicting category labels,
   broken down by the categories involved;
8. token-length distribution of the serialized rows under the fine-tuned
   checkpoint's WordPiece tokenizer (via the `tokenizers` library, torch-free);
9. re-derivation of the DistilBERT-vs-baseline paired bootstrap for each seed
   with the resample count and RNG seed of scripts/19 (4,000; seed 42), to
   confirm the resample count behind the reported intervals, plus
   Bonferroni-adjusted (6 comparisons) 99.17% intervals and McNemar p-values.

Torch-free; seeded; runs in a few minutes (dominated by check 9).
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

from pipeline_config import CLASSES, FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "data" / "revision"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REV / "results" / "revision_quick_checks")
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-bootstrap", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = {}
    result = {}

    splits = {}
    for split in ("train", "val", "test"):
        path = REV / "splits" / f"{split}.parquet"
        inputs[f"{split}_split"] = path
        splits[split] = pd.read_parquet(path)

    # 1. BruteForce / SSH
    cohort_path = REV / "xai" / "cohorts" / "original_xgb.parquet"
    inputs["xai_cohort"] = cohort_path
    cohort = pd.read_parquet(cohort_path).merge(splits["test"][["SampleID", "SSH", "Category"]], on="SampleID")
    ssh = {}
    for split in ("train", "test"):
        frame = splits[split]
        bf = frame["Category"] == "BruteForce"
        ssh[split] = {
            "bruteforce_rows": int(bf.sum()),
            "bruteforce_ssh_positive": float((frame.loc[bf, "SSH"] > 0).mean()),
            "other_rows_ssh_positive": float((frame.loc[~bf, "SSH"] > 0).mean()),
            "benign_ssh_positive": float((frame.loc[frame["Category"] == "Benign", "SSH"] > 0).mean()),
            "share_of_ssh_positive_rows_that_are_bruteforce": float(bf[frame["SSH"] > 0].mean()),
        }
    bfc = cohort[cohort["Category"] == "BruteForce"]
    ssh["xai_cohort"] = {"bruteforce_rows": int(len(bfc)), "bruteforce_ssh_positive": float((bfc["SSH"] > 0).mean())}
    result["bruteforce_ssh"] = ssh

    # 2. Recon composition
    recon = splits["test"].loc[splits["test"]["Category"] == "Recon", "Label"].value_counts()
    result["recon_test_composition"] = {k: {"n": int(v), "share": float(v / recon.sum())} for k, v in recon.items()}

    # 3. batching speed-up
    bench_path = REV / "results" / "cpu_benchmark" / "cpu_benchmark_summary.csv"
    inputs["cpu_benchmark"] = bench_path
    bench = pd.read_csv(bench_path)
    speed = {}
    for mode in ("model_only", "with_input_preparation"):
        sub = bench[bench["mode"] == mode].pivot(index="model", columns="batch_size", values="median_ms_per_sample")
        speed[mode] = {m: float(sub.loc[m, 1] / sub.loc[m, 64]) for m in sub.index}
    result["batch1_over_batch64_speedup"] = speed

    # 4. IG convergence delta
    captum_path = REV / "xai" / "main" / "distilbert" / "captum_per_sample.parquet"
    inputs["captum_per_sample"] = captum_path
    delta = pd.read_parquet(captum_path)["convergence_delta"].abs()
    result["ig_convergence_delta_abs"] = {
        "n": int(len(delta)), "mean": float(delta.mean()), "median": float(delta.median()),
        "p95": float(delta.quantile(0.95)), "max": float(delta.max()),
    }

    # 5. McNemar no_number
    mcn_path = REV / "results" / "distilbert_original_vs_no_number_mcnemar.csv"
    inputs["no_number_mcnemar"] = mcn_path
    result["no_number_mcnemar_p"] = {int(r["seed"]): float(r["exact_p"]) for r in pd.read_csv(mcn_path).to_dict("records")}

    # 6. per-class loss concentration
    pc_path = REV / "results" / "distilbert_original_vs_no_number_per_class_summary.csv"
    inputs["no_number_per_class"] = pc_path
    pc = pd.read_csv(pc_path)
    delta_col = next(c for c in pc.columns if "delta" in c and "mean" in c)
    losses = pc.loc[pc[delta_col] < 0].set_index(pc.columns[0])[delta_col]
    result["no_number_loss_concentration"] = {
        "summed_negative_delta": float(losses.sum()),
        "web_plus_bruteforce_share": float(losses.loc[["Web", "BruteForce"]].sum() / losses.sum()),
        "per_class_delta": {k: float(v) for k, v in pc.set_index(pc.columns[0])[delta_col].items()},
    }

    # 7. label-conflict duplicate groups
    full = pd.concat([splits[s].assign(split=s) for s in splits], ignore_index=True)
    full["group"] = full.groupby(FEATURE_COLUMNS, dropna=False, sort=False).ngroup()
    sizes = full.groupby("group").size()
    dup = full[full["group"].isin(sizes[sizes > 1].index)]
    cats = dup.groupby("group")["Category"].agg(lambda s: tuple(sorted(set(s))))
    conflict = cats[cats.map(len) > 1]
    pair_counts = conflict.map(lambda t: " / ".join(t)).value_counts()
    rows_in_conflict = dup[dup["group"].isin(conflict.index)]
    result["label_conflicts"] = {
        "rows_total": int(len(full)),
        "duplicate_groups": int((sizes > 1).sum()),
        "duplicate_rows": int(len(dup)),
        "conflicting_category_groups": int(len(conflict)),
        "rows_in_conflicting_category_groups": int(len(rows_in_conflict)),
        "category_combinations": {k: int(v) for k, v in pair_counts.items()},
        "rows_in_conflicting_groups_by_category": {k: int(v) for k, v in rows_in_conflict["Category"].value_counts().items()},
    }

    # 7b. capture-file overlap between partitions
    train_files = set(splits["train"]["SourceFile"])
    result["source_file_overlap"] = {
        "distinct_files_train": int(len(train_files)),
        "distinct_files_test": int(splits["test"]["SourceFile"].nunique()),
        "test_rows_from_files_also_in_train": float(splits["test"]["SourceFile"].isin(train_files).mean()),
    }

    # 8. token lengths (tokenizers only; no torch)
    from tokenizers import Tokenizer
    tok_path = REV / "runs" / "distilbert" / "original" / "seed_123" / "model" / "tokenizer.json"
    inputs["tokenizer"] = tok_path
    tokenizer = Tokenizer.from_file(str(tok_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    lengths = {}
    for split in ("train", "val", "test"):
        path = REV / "serialized" / f"{split}.parquet"
        inputs[f"serialized_{split}"] = path
        texts = pd.read_parquet(path)["text"].tolist()
        n = np.array([len(e.ids) for e in tokenizer.encode_batch(texts)])
        lengths[split] = {"n": int(len(n)), "median": float(np.median(n)), "p99": float(np.percentile(n, 99)),
                          "max": int(n.max()), "rows_over_224": int((n > 224).sum())}
    result["token_lengths_with_special_tokens"] = lengths

    # 9. paired bootstrap re-derivation + Bonferroni
    if not args.skip_bootstrap:
        base_dir = REV / "runs" / "baselines" / "original"
        boot = {}
        for seed in (42, 123, 2026):
            path = REV / "runs" / "distilbert" / "original" / f"seed_{seed}" / "test_predictions.parquet"
            inputs[f"distilbert_seed_{seed}_predictions"] = path
            merged = pd.read_parquet(path)[["SampleID", "y_true", "y_pred"]].rename(columns={"y_pred": "db"})
            for key in ("xgb", "rf"):
                frame = pd.read_parquet(base_dir / f"test_predictions_{key}.parquet")[["SampleID", "y_pred"]]
                merged = merged.merge(frame.rename(columns={"y_pred": key}), on="SampleID", validate="one_to_one")
            y = merged["y_true"].to_numpy()
            rng = np.random.default_rng(args.seed)
            n = len(y)
            diffs = {"xgb": np.empty(args.bootstrap_resamples), "rf": np.empty(args.bootstrap_resamples)}
            preds = {m: merged[m].to_numpy() for m in ("db", "xgb", "rf")}
            for i in range(args.bootstrap_resamples):
                idx = rng.integers(0, n, n)
                values = {m: f1_score(y[idx], p[idx], average="macro") for m, p in preds.items()}
                diffs["xgb"][i] = values["db"] - values["xgb"]
                diffs["rf"][i] = values["db"] - values["rf"]
            entry = {}
            for key in ("xgb", "rf"):
                a_ok, b_ok = preds["db"] == y, preds[key] == y
                a_only, b_only = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
                p = float(binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue)
                entry[key] = {
                    "ci95": [float(np.percentile(diffs[key], 2.5)), float(np.percentile(diffs[key], 97.5))],
                    "ci_bonferroni_99.17": [float(np.percentile(diffs[key], 100 * 0.05 / 12)),
                                            float(np.percentile(diffs[key], 100 - 100 * 0.05 / 12))],
                    "mcnemar_p": p, "mcnemar_p_bonferroni6": min(1.0, 6 * p),
                }
            boot[seed] = entry
        result["paired_bootstrap_rederived"] = {"resamples": args.bootstrap_resamples, "rng_seed": args.seed, "by_seed": boot}

    json_path = out / "revision_quick_checks.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# Revision quick checks", ""]
    s = result["bruteforce_ssh"]
    lines += ["## 1. SSH in BruteForce windows", "",
              f"- test: {s['test']['bruteforce_ssh_positive']:.1%} of {s['test']['bruteforce_rows']} BruteForce windows have SSH > 0 "
              f"(other categories {s['test']['other_rows_ssh_positive']:.1%}; Benign {s['test']['benign_ssh_positive']:.1%}); "
              f"{s['test']['share_of_ssh_positive_rows_that_are_bruteforce']:.1%} of SSH-positive test windows are BruteForce.",
              f"- train: {s['train']['bruteforce_ssh_positive']:.1%} of {s['train']['bruteforce_rows']} (other {s['train']['other_rows_ssh_positive']:.1%}).",
              f"- XAI cohort: {s['xai_cohort']['bruteforce_ssh_positive']:.1%} of {s['xai_cohort']['bruteforce_rows']} BruteForce rows.", ""]
    lines += ["## 2. Recon composition (test)", ""] + [
        f"- {k}: {v['n']} ({v['share']:.1%})" for k, v in result["recon_test_composition"].items()] + [""]
    lines += ["## 3. Batch-1 / batch-64 per-sample latency ratio", "", "| Model | model_only | with_input_preparation |", "|---|---:|---:|"]
    for m in speed["model_only"]:
        lines.append(f"| {m} | {speed['model_only'][m]:.2f}x | {speed['with_input_preparation'][m]:.2f}x |")
    d = result["ig_convergence_delta_abs"]
    lines += ["", "## 4. Layer-IG |convergence delta| (logit units)", "",
              f"n = {d['n']}; mean {d['mean']:.4f}; median {d['median']:.4f}; p95 {d['p95']:.4f}; max {d['max']:.4f}.", ""]
    lines += ["## 5. McNemar, no_number vs original", ""] + [
        f"- seed {k}: p = {v:.4f}" for k, v in result["no_number_mcnemar_p"].items()] + [""]
    c = result["no_number_loss_concentration"]
    lines += ["## 6. Concentration of the no_number per-class F1 loss", "",
              f"Summed negative mean per-class deltas {c['summed_negative_delta']:.4f}; Web + BruteForce share {c['web_plus_bruteforce_share']:.1%}.", ""]
    lc = result["label_conflicts"]
    lines += ["## 7. Exact-feature duplicate groups with conflicting categories (all 59,997 rows)", "",
              f"- duplicate groups {lc['duplicate_groups']} ({lc['duplicate_rows']} rows); conflicting-category groups "
              f"{lc['conflicting_category_groups']} ({lc['rows_in_conflicting_category_groups']} rows).",
              "- category combinations: " + "; ".join(f"{k}: {v}" for k, v in lc["category_combinations"].items()) + ".",
              "- rows in conflicting groups by category: " + "; ".join(f"{k}: {v}" for k, v in lc["rows_in_conflicting_groups_by_category"].items()) + ".", ""]
    so = result["source_file_overlap"]
    lines += ["## 7b. Capture-file overlap", "",
              f"{so['test_rows_from_files_also_in_train']:.1%} of test rows come from raw CSV files that also contribute train rows "
              f"({so['distinct_files_test']} files in test, {so['distinct_files_train']} in train); the split is grouped by "
              "serialized content, not by capture file or time.", ""]
    lines += ["## 8. WordPiece token length (incl. [CLS]/[SEP])", "", "| Split | n | median | p99 | max | > 224 |", "|---|---:|---:|---:|---:|---:|"]
    for split, v in lengths.items():
        lines.append(f"| {split} | {v['n']} | {v['median']:.0f} | {v['p99']:.0f} | {v['max']} | {v['rows_over_224']} |")
    if "paired_bootstrap_rederived" in result:
        b = result["paired_bootstrap_rederived"]
        lines += ["", f"## 9. Paired bootstrap re-derived ({b['resamples']:,} resamples, RNG seed {b['rng_seed']})", "",
                  "| Seed | Comparison | 95% CI | Bonferroni 99.17% CI | McNemar p | x6 |", "|---:|---|---|---|---:|---:|"]
        for seed, entry in b["by_seed"].items():
            for key, label in (("xgb", "DistilBERT - XGBoost"), ("rf", "DistilBERT - RF")):
                e = entry[key]
                lines.append(f"| {seed} | {label} | [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] | "
                             f"[{e['ci_bonferroni_99.17'][0]:+.4f}, {e['ci_bonferroni_99.17'][1]:+.4f}] | "
                             f"{e['mcnemar_p']:.2e} | {e['mcnemar_p_bonferroni6']:.2e} |")
    lines.append("")
    md_path = out / "revision_quick_checks.md"
    md_path.write_text("\n".join(lines))
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/40_revision_quick_checks.py",
        "inputs": {k: {"path": str(Path(p).resolve().relative_to(ROOT)), "sha256": sha256(Path(p))} for k, p in inputs.items()},
        "outputs": {p.name: sha256(p) for p in (json_path, md_path)},
    }
    (out / "revision_quick_checks_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
