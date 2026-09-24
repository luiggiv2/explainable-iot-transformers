"""Fine-tune and evaluate DistilBERT with self-contained run provenance.

Each run writes its checkpoint, training curve, FP32 test predictions, class
probabilities, confusion matrix, metrics, and SHA-256 manifest together. This
prevents the previous failure mode where BF16-reported metrics were combined
with a separately regenerated FP32 prediction array.

Training may use BF16 autocast on Apple MPS. Validation, checkpoint selection,
and final inference always use FP32. The revised experiment defaults to at most
15 epochs and early stopping with patience 4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline_config import CLASSES, FEATURE_SET_EXCLUSIONS


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 224
DEFAULT_BATCH = 32
DEFAULT_EVAL_BATCH = 64
DEFAULT_LR = 2e-5
DEFAULT_MAX_EPOCHS = 15
DEFAULT_PATIENCE = 4
WARMUP_FRAC = 0.1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serialized-dir", type=Path, default=DATA / "serialized")
    parser.add_argument("--run-dir", type=Path, default=DATA)
    parser.add_argument("--model-dir", type=Path, default=DATA / "models" / "distilbert")
    parser.add_argument("--feature-set", choices=FEATURE_SET_EXCLUSIONS, default="original")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--fp32-training",
        action="store_true",
        help="Disable BF16 autocast during MPS training; evaluation is always FP32.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a model directory that already contains a checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data/model and execute one FP32 batch without training or saving.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested, but MPS is unavailable")
    return torch.device(requested)


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


class FlowTextDataset(Dataset):
    def __init__(self, path: Path, tokenizer):
        self.path = path.resolve()
        self.frame = pd.read_parquet(path).reset_index(drop=True)
        required = {"text", "Label", "Category"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        self.enc = tokenizer(
            self.frame["text"].tolist(), truncation=True, max_length=MAX_LEN
        )
        self.labels = pd.Categorical(
            self.frame["Category"], categories=CLASSES
        ).codes
        if np.any(self.labels < 0):
            raise ValueError(f"unknown category in {path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.enc["input_ids"][index],
            "attention_mask": self.enc["attention_mask"][index],
            "label": int(self.labels[index]),
        }


class Collator:
    """Pad a batch and stack labels; top-level for macOS worker pickling."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        labels = torch.tensor([item.pop("label") for item in batch])
        padded = self.tokenizer.pad(batch, return_tensors="pt")
        padded["labels"] = labels
        return padded


def synchronize(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def predict_fp32(model, loader, device: torch.device):
    """Return logits and labels without autocast for all reported evaluation."""
    model.eval()
    logits, labels = [], []
    for batch in loader:
        y = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        logits.append(output.logits.float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def metric_rows(report, auc, train_time, infer_ms, infer_ms_bs1, model_size_mb):
    rows = []
    for cls in CLASSES:
        values = report[cls]
        rows.append(
            {
                "model": "distilbert",
                "class": cls,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": int(values["support"]),
            }
        )
    rows.append({"model": "distilbert", "class": "_accuracy", "f1": report["accuracy"]})
    for aggregate in ("macro avg", "weighted avg"):
        values = report[aggregate]
        rows.append(
            {
                "model": "distilbert",
                "class": f"_{aggregate.replace(' ', '_')}",
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
            }
        )
    rows.extend(
        [
            {"model": "distilbert", "class": "_roc_auc_macro_ovr", "f1": auc},
            {"model": "distilbert", "class": "_train_time_s", "f1": train_time},
            {
                "model": "distilbert",
                "class": "_diagnostic_infer_ms_per_sample",
                "f1": infer_ms,
            },
            {
                "model": "distilbert",
                "class": "_diagnostic_infer_ms_per_sample_bs1",
                "f1": infer_ms_bs1,
            },
            {"model": "distilbert", "class": "_model_size_mb", "f1": model_size_mb},
        ]
    )
    return rows


def main():
    args = parse_args()
    serialized_dir = args.serialized_dir.resolve()
    run_dir = args.run_dir.resolve()
    model_dir = args.model_dir.resolve()
    checkpoint = model_dir / "model.safetensors"
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(
            f"checkpoint already exists at {checkpoint}; pass --overwrite explicitly"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(args.device)
    use_bf16_train = device.type == "mps" and not args.fp32_training

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = Collator(tokenizer)
    train_ds = FlowTextDataset(serialized_dir / "train.parquet", tokenizer)
    val_ds = FlowTextDataset(serialized_dir / "val.parquet", tokenizer)
    test_ds = FlowTextDataset(serialized_dir / "test.parquet", tokenizer)
    print(
        f"train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)} | "
        f"device {device} | feature_set {args.feature_set} | seed {seed}"
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.eval_batch_size, collate_fn=collate
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASSES),
        id2label=dict(enumerate(CLASSES)),
        label2id={name: index for index, name in enumerate(CLASSES)},
    ).to(device)
    if args.dry_run:
        dry_loader = DataLoader(
            [val_ds[index] for index in range(min(args.eval_batch_size, len(val_ds)))],
            batch_size=args.eval_batch_size,
            collate_fn=collate,
        )
        dry_logits, dry_y = predict_fp32(model, dry_loader, device)
        assert dry_logits.shape == (len(dry_y), len(CLASSES))
        print(
            f"dry run passed: FP32 logits {tuple(dry_logits.shape)}, "
            f"labels {tuple(dry_y.shape)}"
        )
        return

    total_steps = len(train_loader) * args.max_epochs
    warmup_steps = int(total_steps * WARMUP_FRAC)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: step / max(1, warmup_steps)
        if step < warmup_steps
        else max(
            0.0,
            (total_steps - step) / max(1, total_steps - warmup_steps),
        ),
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = []
    train_started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        epoch_started = time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=use_bf16_train
            ):
                output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_sum += output.loss.detach().float()
            if step % 200 == 0:
                rate = step / (time.perf_counter() - epoch_started)
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss {loss_sum.item() / step:.4f} ({rate:.2f} it/s)",
                    flush=True,
                )
        train_loss = loss_sum.item() / len(train_loader)
        val_logits, val_y = predict_fp32(model, val_loader, device)
        val_f1 = f1_score(val_y.numpy(), val_logits.argmax(1).numpy(), average="macro")
        elapsed = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_f1_fp32": val_f1,
                "epoch_time_s": elapsed,
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "distilbert_training_log.csv", index=False)
        print(
            f"epoch {epoch}: loss={train_loss:.4f} "
            f"val_macroF1_fp32={val_f1:.4f} ({elapsed:.0f}s)",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            model.save_pretrained(model_dir, state_dict=best_state)
            tokenizer.save_pretrained(model_dir)
            print(f"  checkpoint saved (epoch {epoch})", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"early stop at epoch {epoch} (best epoch {best_epoch})",
                    flush=True,
                )
                break
    train_time = time.perf_counter() - train_started
    if best_state is None:
        raise RuntimeError("training completed without a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()

    # FP32 test inference is the single source for metrics and downstream XAI.
    synchronize(device)
    inference_started = time.perf_counter()
    test_logits, test_y = predict_fp32(model, test_loader, device)
    synchronize(device)
    infer_ms = (time.perf_counter() - inference_started) / len(test_ds) * 1e3

    single_count = min(200, len(test_ds))
    single_loader = DataLoader(
        [test_ds[index] for index in range(single_count)],
        batch_size=1,
        collate_fn=collate,
    )
    warmup_loader = DataLoader(
        [test_ds[index] for index in range(min(5, single_count))],
        batch_size=1,
        collate_fn=collate,
    )
    predict_fp32(model, warmup_loader, device)
    synchronize(device)
    single_started = time.perf_counter()
    predict_fp32(model, single_loader, device)
    synchronize(device)
    infer_ms_bs1 = (time.perf_counter() - single_started) / single_count * 1e3

    y_true = test_y.numpy()
    logits = test_logits.numpy()
    y_pred = logits.argmax(axis=1)
    probabilities = torch.softmax(test_logits, dim=1).numpy()
    np.save(run_dir / "distilbert_test_preds.npy", y_pred)

    records = pd.DataFrame(
        {
            "row": np.arange(len(test_ds)),
            "subtype": test_ds.frame["Label"].to_numpy(),
            "true_category": test_ds.frame["Category"].to_numpy(),
            "predicted_category": np.asarray(CLASSES, dtype=object)[y_pred],
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    if "SampleID" in test_ds.frame:
        records.insert(1, "SampleID", test_ds.frame["SampleID"].to_numpy())
    for index, cls in enumerate(CLASSES):
        records[f"prob_{cls}"] = probabilities[:, index]
    records.to_parquet(run_dir / "test_predictions.parquet", index=False)

    auc = roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(
        run_dir / "confusion_distilbert.csv"
    )

    model_size_mb = sum(
        path.stat().st_size for path in model_dir.rglob("*") if path.is_file()
    ) / 1e6
    rows = metric_rows(
        report, auc, train_time, infer_ms, infer_ms_bs1, model_size_mb
    )
    pd.DataFrame(rows).to_csv(run_dir / "transformer_metrics.csv", index=False)

    reconstructed = confusion_matrix(
        records["y_true"], records["y_pred"], labels=np.arange(len(CLASSES))
    )
    assert np.array_equal(matrix, reconstructed)
    assert np.array_equal(np.load(run_dir / "distilbert_test_preds.npy"), y_pred)
    checkpoint_hash = sha256(checkpoint)
    prediction_hash = sha256(run_dir / "test_predictions.parquet")
    split_hash = sha256(serialized_dir / "test.parquet")

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_name": MODEL_NAME,
        "feature_set": args.feature_set,
        "seed": seed,
        "classes": CLASSES,
        "serialized_dir": str(serialized_dir.relative_to(ROOT)),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "model_dir": str(model_dir.relative_to(ROOT)),
        "train_precision": "bf16_autocast" if use_bf16_train else "fp32",
        "validation_precision": "fp32",
        "test_precision": "fp32",
        "device": device.type,
        "max_length": MAX_LEN,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "warmup_fraction": WARMUP_FRAC,
        "best_epoch": best_epoch,
        "best_val_macro_f1_fp32": best_f1,
        "test_macro_f1_fp32": report["macro avg"]["f1-score"],
        "test_accuracy_fp32": report["accuracy"],
        "rows": {
            "train": len(train_ds),
            "validation": len(val_ds),
            "test": len(test_ds),
        },
        "sha256": {
            "serialized_test": split_hash,
            "checkpoint": checkpoint_hash,
            "test_predictions": prediction_hash,
        },
        "consistency_checks": {
            "prediction_array_equal": True,
            "confusion_reconstructed_equal": True,
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    metric_frame = pd.DataFrame(rows)
    per_class = metric_frame.query("~`class`.str.startswith('_')")
    aggregate = metric_frame.query("`class`.str.startswith('_')")
    summary = [
        "# DistilBERT fine-tuning — CICIoT2023 8-class",
        "",
        f"Feature set `{args.feature_set}`; seed {seed}; {MODEL_NAME}; max_len "
        f"{MAX_LEN}; batch {args.batch_size}; lr {args.learning_rate}; training "
        f"precision {'BF16 autocast' if use_bf16_train else 'FP32'}; validation and "
        f"test FP32; best epoch {best_epoch}/{args.max_epochs} (validation macro-F1 "
        f"{best_f1:.4f}); early-stopping patience {args.patience}; device {device.type}.",
        "",
        "## Per-class metrics",
        per_class.set_index("class")[["precision", "recall", "f1", "support"]]
        .round(4)
        .to_markdown(),
        "",
        "## Aggregates and diagnostic cost",
        aggregate.set_index("class")[["f1"]].round(4).to_markdown(),
        "",
        "All reported metrics, the confusion matrix, and downstream statistics must "
        "use `test_predictions.parquet` from this same run. Latencies here are "
        "diagnostic; the controlled RQ5 benchmark is reported separately.",
        "",
    ]
    (run_dir / "transformer_summary.md").write_text("\n".join(summary))
    print(
        f"distilbert: macroF1={report['macro avg']['f1-score']:.4f} "
        f"acc={report['accuracy']:.4f} auc={auc:.4f} best_epoch={best_epoch} "
        f"checkpoint_sha256={checkpoint_hash[:12]}...",
        flush=True,
    )
    print(f"wrote complete run to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
