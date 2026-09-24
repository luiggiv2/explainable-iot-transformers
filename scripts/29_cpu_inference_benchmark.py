"""Controlled CPU inference benchmark for RQ5.

All four fitted models are evaluated on the same fixed, class-balanced rows,
under one explicit CPU thread budget. Timings separate model-only inference
from input preparation and retain every repeated measurement.

PyTorch and XGBoost each load their own OpenMP runtime, and on macOS the two
cannot coexist in one process: the combination either deadlocks or raises
SIGSEGV inside the OpenMP fork barrier. The benchmark therefore measures each
family in a separate child process (`--models distilbert` and
`--models classic`) and merges the partial results. Because the two children
run sequentially and alone, this also removes any cross-library thread
interference from the measurement itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pipeline_config import CLASSES, feature_pairs


MAX_LEN = 224
FAMILIES = ("distilbert", "classic")
CLASSIC_MODELS = ("rf", "svm", "xgb")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--serialized-test", type=Path, required=True)
    parser.add_argument("--distilbert-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 64))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", choices=("all",) + FAMILIES, default="all")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size_mb(path: Path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1e6


def batches(length, batch_size):
    for start in range(0, length, batch_size):
        yield start, min(start + batch_size, length)


def timed_repetitions(function, warmups, repetitions):
    for _ in range(warmups):
        function()
    samples = []
    for repetition in range(repetitions):
        started = time.perf_counter_ns()
        function()
        elapsed = (time.perf_counter_ns() - started) / 1e9
        samples.append((repetition + 1, elapsed))
    return samples


def partial_paths(output_dir: Path, family: str):
    return (
        output_dir / f"cpu_benchmark_repetitions__{family}.csv",
        output_dir / f"cpu_benchmark_meta__{family}.json",
    )


def build_cohort(args, output_dir: Path):
    """Return the shared benchmark cohort, creating it once if absent.

    Both children must time exactly the same rows in the same order, so the
    slim cohort file is the single source of truth once it exists.
    """
    columns = [column for column, _ in feature_pairs("original")]
    split = pd.read_parquet(args.test_split).reset_index(names="row")
    serialized = pd.read_parquet(args.serialized_test).reset_index(
        names="serialized_row"
    )
    distilbert_predictions = pd.read_parquet(args.distilbert_predictions)
    aligned = split.merge(
        serialized[["serialized_row", "SampleID", "Category", "text"]],
        left_on=["row", "SampleID", "Category"],
        right_on=["serialized_row", "SampleID", "Category"],
        validate="one_to_one",
    ).merge(
        distilbert_predictions[["row", "SampleID", "y_pred"]],
        on=["row", "SampleID"],
        validate="one_to_one",
    )
    if len(aligned) != len(split):
        raise ValueError("test split, serialization, and predictions did not align")

    cohort_path = output_dir / "cpu_benchmark_cohort.parquet"
    if cohort_path.exists():
        slim = pd.read_parquet(cohort_path)
    else:
        rng = np.random.default_rng(args.seed)
        chosen = []
        for class_name in CLASSES:
            pool = aligned[aligned["Category"] == class_name]
            if len(pool) < args.per_class:
                raise ValueError(
                    f"{class_name} has fewer than {args.per_class} test rows"
                )
            selected_indices = rng.choice(
                pool.index.to_numpy(), args.per_class, replace=False
            )
            chosen.append(aligned.loc[selected_indices])
        slim = pd.concat(chosen).reset_index(drop=True)
        slim.insert(0, "cohort_order", np.arange(len(slim), dtype=np.int64))
        slim = slim[["cohort_order", "row", "SampleID", "Category"]]
        slim.to_parquet(cohort_path, index=False)

    counts = slim.groupby("Category").size().to_dict()
    if any(counts.get(name) != args.per_class for name in CLASSES):
        raise ValueError("saved benchmark cohort does not match the requested design")
    cohort = slim.merge(
        aligned, on=["row", "SampleID", "Category"], validate="one_to_one"
    ).sort_values("cohort_order")
    return cohort, columns, cohort_path


def expected_classic(cohort, baseline_dir: Path, name: str):
    predictions = pd.read_parquet(baseline_dir / f"test_predictions_{name}.parquet")
    match = (
        cohort[["cohort_order", "row", "SampleID"]]
        .merge(
            predictions[["row", "SampleID", "y_pred"]],
            on=["row", "SampleID"],
            validate="one_to_one",
        )
        .sort_values("cohort_order")
    )
    return match["y_pred"].to_numpy()


def run_distilbert(args, cohort, output_dir: Path):
    """Time the encoder alone; torch is imported only in this child process."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = args.model_dir.resolve()
    texts = cohort["text"].tolist()
    expected = cohort["y_pred"].to_numpy()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    transformer = AutoModelForSequenceClassification.from_pretrained(model_dir).cpu()
    transformer.eval()
    encodings = [
        tokenizer(text, truncation=True, max_length=MAX_LEN, add_special_tokens=True)
        for text in texts
    ]

    rows = []
    validation = {}
    with threadpool_limits(limits=args.threads):
        for batch_size in args.batch_sizes:
            prepared_batches = [
                tokenizer.pad(encodings[start:end], padding=True, return_tensors="pt")
                for start, end in batches(len(cohort), batch_size)
            ]

            @torch.inference_mode()
            def model_only():
                output = []
                for encoded in prepared_batches:
                    output.extend(transformer(**encoded).logits.argmax(1).tolist())
                return np.asarray(output)

            @torch.inference_mode()
            def with_tokenization():
                output = []
                for start, end in batches(len(cohort), batch_size):
                    encoded = tokenizer(
                        texts[start:end],
                        truncation=True,
                        max_length=MAX_LEN,
                        padding=True,
                        return_tensors="pt",
                    )
                    output.extend(transformer(**encoded).logits.argmax(1).tolist())
                return np.asarray(output)

            validation[str(batch_size)] = bool(
                np.array_equal(model_only(), expected)
            )
            for mode, function in (
                ("model_only", model_only),
                ("with_input_preparation", with_tokenization),
            ):
                for repetition, seconds in timed_repetitions(
                    function, args.warmup_runs, args.repetitions
                ):
                    rows.append(
                        {
                            "model": "distilbert",
                            "batch_size": batch_size,
                            "mode": mode,
                            "repetition": repetition,
                            "seconds": seconds,
                            "ms_per_sample": seconds / len(cohort) * 1000,
                            "samples_per_second": len(cohort) / seconds,
                        }
                    )
            print(f"completed distilbert, batch {batch_size}", flush=True)

    meta = {
        "family": "distilbert",
        "torch": torch.__version__,
        "validation": validation,
        "sizes_mb": {"distilbert": directory_size_mb(model_dir)},
        "input_sha256": {
            "checkpoint": sha256(model_dir / "model.safetensors"),
        },
    }
    return pd.DataFrame(rows), meta


def run_classic(args, cohort, columns, output_dir: Path):
    """Time RF, SVM and XGBoost; this child never imports torch."""
    baseline_dir = args.baseline_dir.resolve()
    numeric = cohort[columns].to_numpy(np.float32)

    estimators = {}
    for name in CLASSIC_MODELS:
        path = baseline_dir / "models" / f"{name}.joblib"
        estimator = joblib.load(path)
        if hasattr(estimator, "set_params"):
            try:
                estimator.set_params(n_jobs=args.threads)
            except ValueError:
                pass
        estimators[name] = (estimator, path)

    rows = []
    validation = {}
    with threadpool_limits(limits=args.threads):
        for batch_size in args.batch_sizes:
            for name, (estimator, _) in estimators.items():
                expected = expected_classic(cohort, baseline_dir, name)

                def model_only(estimator=estimator):
                    output = []
                    for start, end in batches(len(cohort), batch_size):
                        output.extend(estimator.predict(numeric[start:end]).tolist())
                    return np.asarray(output)

                def with_preparation(estimator=estimator):
                    output = []
                    for start, end in batches(len(cohort), batch_size):
                        values = cohort.iloc[start:end][columns].to_numpy(np.float32)
                        output.extend(estimator.predict(values).tolist())
                    return np.asarray(output)

                validation[f"{name}:{batch_size}"] = bool(
                    np.array_equal(model_only(), expected)
                )
                for mode, function in (
                    ("model_only", model_only),
                    ("with_input_preparation", with_preparation),
                ):
                    for repetition, seconds in timed_repetitions(
                        function, args.warmup_runs, args.repetitions
                    ):
                        rows.append(
                            {
                                "model": name,
                                "batch_size": batch_size,
                                "mode": mode,
                                "repetition": repetition,
                                "seconds": seconds,
                                "ms_per_sample": seconds / len(cohort) * 1000,
                                "samples_per_second": len(cohort) / seconds,
                            }
                        )
                print(f"completed {name}, batch {batch_size}", flush=True)

    meta = {
        "family": "classic",
        "validation": validation,
        "sizes_mb": {
            name: path.stat().st_size / 1e6 for name, (_, path) in estimators.items()
        },
        "input_sha256": {
            name: sha256(path) for name, (_, path) in estimators.items()
        },
    }
    return pd.DataFrame(rows), meta


def child_command(args, family: str):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-dir", str(args.model_dir),
        "--baseline-dir", str(args.baseline_dir),
        "--test-split", str(args.test_split),
        "--serialized-test", str(args.serialized_test),
        "--distilbert-predictions", str(args.distilbert_predictions),
        "--output-dir", str(args.output_dir),
        "--per-class", str(args.per_class),
        "--batch-sizes", *[str(size) for size in args.batch_sizes],
        "--threads", str(args.threads),
        "--warmup-runs", str(args.warmup_runs),
        "--repetitions", str(args.repetitions),
        "--seed", str(args.seed),
        "--models", family,
    ]


def finalize(args, output_dir: Path, cohort_path: Path, cohort_rows: int):
    frames, metas = [], {}
    for family in FAMILIES:
        csv_path, meta_path = partial_paths(output_dir, family)
        if not csv_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"missing partial results for {family}")
        frames.append(pd.read_csv(csv_path))
        metas[family] = json.loads(meta_path.read_text())

    validation = {
        family: metas[family]["validation"] for family in FAMILIES
    }
    failures = [
        f"{family}:{key}"
        for family, checks in validation.items()
        for key, passed in checks.items()
        if not passed
    ]
    if failures:
        raise ValueError(f"benchmark predictions differ from saved artifacts: {failures}")

    raw = pd.concat(frames, ignore_index=True)
    raw_path = output_dir / "cpu_benchmark_repetitions.csv"
    raw.to_csv(raw_path, index=False)

    summary = (
        raw.groupby(["model", "batch_size", "mode"], as_index=False)
        .agg(
            median_ms_per_sample=("ms_per_sample", "median"),
            q1_ms_per_sample=("ms_per_sample", lambda values: values.quantile(0.25)),
            q3_ms_per_sample=("ms_per_sample", lambda values: values.quantile(0.75)),
            median_samples_per_second=("samples_per_second", "median"),
        )
        .sort_values(["batch_size", "mode", "median_ms_per_sample"])
    )
    sizes = {}
    for family in FAMILIES:
        sizes.update(metas[family]["sizes_mb"])
    summary["model_size_mb"] = summary["model"].map(sizes)
    summary_path = output_dir / "cpu_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    lines = [
        "# Controlled CPU inference benchmark",
        "",
        f"All models use the same {cohort_rows} rows, {args.threads} CPU threads, "
        f"{args.warmup_runs} warm-up pass(es), and {args.repetitions} measured "
        "passes. Values are medians across measured passes; brackets show the "
        "interquartile range.",
        "",
        "| Model | Batch | Timing boundary | ms/sample | samples/s | Size (MB) |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.batch_size} | {row.mode} | "
            f"{row.median_ms_per_sample:.4f} "
            f"[{row.q1_ms_per_sample:.4f}, {row.q3_ms_per_sample:.4f}] | "
            f"{row.median_samples_per_second:.1f} | {row.model_size_mb:.2f} |"
        )
    lines.extend(
        [
            "",
            "`model_only` excludes tokenization or DataFrame-to-array conversion. "
            "`with_input_preparation` includes those operations but assumes that a "
            "serialized flow string or numeric feature row already exists. These "
            "measurements characterize this machine and software stack; they are not "
            "hardware-independent latency claims.",
            "",
            "The encoder and the classical models were timed in separate, "
            "sequentially executed processes because PyTorch and XGBoost cannot "
            "share one process on this platform. Each family therefore held the "
            "declared thread budget alone, and no measurement overlapped another.",
            "",
        ]
    )
    markdown_path = output_dir / "cpu_benchmark_summary.md"
    markdown_path.write_text("\n".join(lines))

    input_hashes = {"models": {}}
    for family in FAMILIES:
        for key, value in metas[family]["input_sha256"].items():
            if key == "checkpoint":
                input_hashes["checkpoint_sha256"] = value
            else:
                input_hashes["models"][key] = value
    input_hashes.update(
        {
            "test_split_sha256": sha256(args.test_split.resolve()),
            "serialized_test_sha256": sha256(args.serialized_test.resolve()),
            "distilbert_predictions_sha256": sha256(
                args.distilbert_predictions.resolve()
            ),
            "cohort_sha256": sha256(cohort_path),
        }
    )

    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "torch": metas["distilbert"]["torch"],
        "threads": args.threads,
        "warmup_runs": args.warmup_runs,
        "repetitions": args.repetitions,
        "batch_sizes": list(args.batch_sizes),
        "cohort_rows": cohort_rows,
        "process_isolation": {
            "reason": "PyTorch and XGBoost load incompatible OpenMP runtimes on macOS",
            "families": list(FAMILIES),
            "executed": "sequentially, one family per process",
        },
        "all_predictions_match_saved": True,
        "validation": validation,
        "inputs": input_hashes,
        "outputs": {
            "repetitions_sha256": sha256(raw_path),
            "summary_sha256": sha256(summary_path),
            "markdown_sha256": sha256(markdown_path),
        },
    }
    (output_dir / "cpu_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("\n".join(lines))


def main():
    args = parse_args()
    if (
        args.per_class <= 0
        or args.threads <= 0
        or args.warmup_runs < 0
        or args.repetitions <= 0
        or any(size <= 0 for size in args.batch_sizes)
    ):
        raise ValueError("benchmark counts, batches, and thread budget are invalid")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "cpu_benchmark_repetitions.csv"
    if raw_path.exists():
        raise FileExistsError(f"completed benchmark exists at {raw_path}")

    cohort, columns, cohort_path = build_cohort(args, output_dir)

    if args.models == "all":
        for family in FAMILIES:
            csv_path, _ = partial_paths(output_dir, family)
            if csv_path.exists():
                print(f"reusing existing partial results for {family}", flush=True)
                continue
            print(f"--- timing {family} in a dedicated process ---", flush=True)
            completed = subprocess.run(child_command(args, family), check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{family} benchmark process exited with {completed.returncode}"
                )
        finalize(args, output_dir, cohort_path, len(cohort))
        return

    if args.models == "distilbert":
        frame, meta = run_distilbert(args, cohort, output_dir)
    else:
        frame, meta = run_classic(args, cohort, columns, output_dir)

    csv_path, meta_path = partial_paths(output_dir, args.models)
    frame.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote partial benchmark results for {args.models}", flush=True)


if __name__ == "__main__":
    main()
