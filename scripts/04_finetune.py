"""Phase 3 — DistilBERT fine-tuning on textualized IoT flows.

8-class task (Category) over the serialized text of the stratified subset
(same rows and 70/15/15 splits as the classical baselines, so RQ1 is a
like-for-like comparison). distilbert-base-uncased + sequence-classification
head, trained on Apple MPS: batch 32, lr 2e-5, linear warmup/decay, up to
5 epochs with early stopping on validation macro-F1 (patience 2).

Metrics mirror the baseline protocol: per-class F1, macro/weighted
aggregates, ROC-AUC (macro OvR), full confusion matrix, train time,
per-sample inference latency (batched and batch=1, both feed RQ5) and
model size on disk.

Outputs (all under data/):
  transformer_metrics.csv        long table: model x class + aggregate rows
  confusion_distilbert.csv       test confusion matrix (rows=true, cols=pred)
  distilbert_training_log.csv    per-epoch train loss / val macro-F1
  transformer_summary.md         human-readable summary
  models/distilbert/             best checkpoint (reused for Captum in Phase 4)
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_DIR = DATA / "models" / "distilbert"

MODEL_NAME = "distilbert-base-uncased"
CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
MAX_LEN = 224          # p99 token length is 213, observed max 223
BATCH = 32
EVAL_BATCH = 64
LR = 2e-5
MAX_EPOCHS = 10       # raised from 5: the 5-epoch run was still improving on val
PATIENCE = 3          # patience 3 tolerates the noisy val curve before stopping
WARMUP_FRAC = 0.1

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# bf16 is native on Apple M3: ~2x arithmetic throughput, no grad scaler needed
USE_BF16 = DEVICE.type == "mps"


class FlowTextDataset(Dataset):
    def __init__(self, split, tokenizer):
        df = pd.read_parquet(DATA / "serialized" / f"{split}.parquet")
        self.enc = tokenizer(df["text"].tolist(), truncation=True,
                             max_length=MAX_LEN)
        self.labels = pd.Categorical(df["Category"], categories=CLASSES).codes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "label": int(self.labels[i])}


class Collator:
    """Pads a batch and stacks its labels. A top-level class (not a closure)
    so it can be pickled for the DataLoader's spawned CPU workers on macOS."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        labels = torch.tensor([b.pop("label") for b in batch])
        padded = self.tokenizer.pad(batch, return_tensors="pt")
        padded["labels"] = labels
        return padded


@torch.no_grad()
def predict(model, loader):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        y = batch.pop("labels")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=USE_BF16):
            out = model(**batch)
        logits.append(out.logits.float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = Collator(tokenizer)
    train_ds = FlowTextDataset("train", tokenizer)
    val_ds = FlowTextDataset("val", tokenizer)
    test_ds = FlowTextDataset("test", tokenizer)
    print(f"train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)} | device {DEVICE}")

    gen = torch.Generator().manual_seed(SEED)
    # CPU workers prepare/pad batches in parallel so the GPU never waits on them
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              generator=gen, collate_fn=collate,
                              num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH, collate_fn=collate)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CLASSES),
        id2label=dict(enumerate(CLASSES)),
        label2id={c: i for i, c in enumerate(CLASSES)},
    ).to(DEVICE)

    total_steps = len(train_loader) * MAX_EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: s / max(1, warmup_steps) if s < warmup_steps
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup_steps)),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0
    history = []
    t_train = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        # loss accumulated on-device; .item() forces a CPU<->GPU sync, so it
        # only happens at the logging points, never per step
        loss_sum = torch.zeros((), device=DEVICE)
        t0 = time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16,
                                enabled=USE_BF16):
                out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_sum += out.loss.detach().float()
            if step % 200 == 0:
                rate = step / (time.perf_counter() - t0)
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {loss_sum.item() / step:.4f} "
                      f"({rate:.2f} it/s)", flush=True)
        train_loss = loss_sum.item() / len(train_loader)

        val_logits, val_y = predict(model, val_loader)
        val_f1 = f1_score(val_y, val_logits.argmax(1), average="macro")
        dt = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_macro_f1": val_f1, "epoch_time_s": dt})
        print(f"epoch {epoch}: loss={train_loss:.4f} "
              f"val_macroF1={val_f1:.4f} ({dt:.0f}s)", flush=True)

        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            # crash-safe: best checkpoint goes to disk every time it improves
            model.save_pretrained(OUT_DIR, state_dict=best_state)
            print(f"  checkpoint saved (epoch {epoch})", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"early stop at epoch {epoch} (best epoch {best_epoch})")
                break
    train_time = time.perf_counter() - t_train

    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(DATA / "distilbert_training_log.csv", index=False)

    # --- test evaluation (timed batched inference, comparable to baselines)
    t0 = time.perf_counter()
    test_logits, test_y = predict(model, test_loader)
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    infer_ms = (time.perf_counter() - t0) / len(test_ds) * 1e3

    # batch=1 latency: the edge-deployment number for RQ5
    model.eval()
    single_loader = DataLoader([test_ds[i] for i in range(200)],
                               batch_size=1, collate_fn=collate)
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in single_loader:
            batch.pop("labels")
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16,
                                enabled=USE_BF16):
                model(**{k: v.to(DEVICE) for k, v in batch.items()})
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    infer_ms_bs1 = (time.perf_counter() - t0) / 200 * 1e3

    y_pred = test_logits.argmax(1).numpy()
    probs = torch.softmax(test_logits, dim=1).numpy()
    auc = roc_auc_score(test_y, probs, multi_class="ovr", average="macro")
    rep = classification_report(test_y, y_pred, target_names=CLASSES,
                                output_dict=True, zero_division=0)

    rows = []
    name = "distilbert"
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
    rows.append({"model": name, "class": "_infer_ms_per_sample_bs1", "f1": infer_ms_bs1})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    size_mb = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()) / 1e6
    rows.append({"model": name, "class": "_model_size_mb", "f1": size_mb})

    cm = confusion_matrix(test_y, y_pred)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        DATA / "confusion_distilbert.csv")
    pd.DataFrame(rows).to_csv(DATA / "transformer_metrics.csv", index=False)

    lines = [
        "# DistilBERT fine-tuning — CICIoT2023 8-class (test split, n=9,000)",
        "",
        f"{MODEL_NAME}, max_len {MAX_LEN}, batch {BATCH}, lr {LR}, "
        f"{'bf16 autocast' if USE_BF16 else 'fp32'}, "
        f"warmup {WARMUP_FRAC:.0%}, early stopping on val macro-F1 "
        f"(patience {PATIENCE}); best epoch {best_epoch} "
        f"(val macro-F1 {best_f1:.4f}); seed {SEED}; device {DEVICE.type}.",
        "",
        "## Per-class F1",
        pd.DataFrame(rows).query("~`class`.str.startswith('_')")
          .set_index("class")[["precision", "recall", "f1", "support"]]
          .round(4).to_markdown(),
        "",
        "## Aggregates / cost",
        pd.DataFrame([r for r in rows if r["class"].startswith("_")])
          .set_index("class")[["f1"]].round(4).to_markdown(),
        "",
        "Confusion matrix: confusion_distilbert.csv. "
        "Training curve: distilbert_training_log.csv.",
        "",
    ]
    (DATA / "transformer_summary.md").write_text("\n".join(lines))
    print(f"\n{name}: macroF1={rep['macro avg']['f1-score']:.4f} "
          f"acc={rep['accuracy']:.4f} auc={auc:.4f} "
          f"train={train_time:.0f}s infer={infer_ms:.3f}ms/sample "
          f"(bs1 {infer_ms_bs1:.2f}ms) size={size_mb:.0f}MB")
    print("wrote data/transformer_metrics.csv + transformer_summary.md")


if __name__ == "__main__":
    main()
