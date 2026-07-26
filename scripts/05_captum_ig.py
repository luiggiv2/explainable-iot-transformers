"""Phase 4 (primary) — Layer Integrated Gradients on the fine-tuned DistilBERT.

Captum LayerIntegratedGradients over the word-embedding layer explains *why*
the model assigns each attack class. Token-level (wordpiece) attributions are
remapped to the 39 semantic flow features (rate, flags, ports, sizes, ...) via
character offsets, so the output speaks RQ2's language and can be cross-checked
against domain knowledge (RQ3).

Protocol:
  - sample N correctly-classified test rows per class (we characterize the
    signal behind *correct* decisions, not the model's mistakes);
  - attribute each toward its (true == predicted) class, baseline = all [PAD]
    (keeping [CLS]/[SEP]); n_steps=50, fp32 for numerically stable attributions;
  - per sample, sum token attributions per feature, then L1-normalize across
    features so every sample contributes comparably;
  - aggregate per class: mean signed contribution (direction) and mean |contrib|
    (magnitude, used for ranking).

Outputs (data/):
  captum_feature_attribution.csv   long: class x feature -> signed_mean, abs_mean, rank
  captum_per_sample.parquet        per-sample normalized feature attributions (reused in Phase 5)
  captum_top_features.md           human-readable top-8 features per class
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = DATA / "models" / "distilbert"

CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
MAX_LEN = 224
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", "120"))
N_STEPS = 50
FIELD_RE = re.compile(r"([A-Za-z_]+)=")   # matches each "key=" -> feature name + position

# IG needs gradients; MPS handles them but can be flaky for attribution — allow
# a CPU override via env. Attribution is not as heavy as training, so CPU is OK.
DEVICE = torch.device(os.environ.get("XAI_DEVICE",
                      "mps" if torch.backends.mps.is_available() else "cpu"))


def feature_spans(text):
    """Return [(feature_name, start_char, end_char), ...] for each key=value field."""
    keys = [(m.group(1), m.start()) for m in FIELD_RE.finditer(text)]
    spans = []
    for i, (name, start) in enumerate(keys):
        end = keys[i + 1][1] if i + 1 < len(keys) else len(text)
        spans.append((name, start, end))
    return spans


def token_to_feature(offsets, spans):
    """Map each token (by char offset) to a feature index, or -1 for special/pad."""
    idx = np.full(len(offsets), -1, dtype=np.int64)
    for t, (s, e) in enumerate(offsets):
        if s == e:            # special token ([CLS]/[SEP]/[PAD]) -> no text span
            continue
        for f, (_, fs, fe) in enumerate(spans):
            if fs <= s < fe:
                idx[t] = f
                break
    return idx


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    print(f"device {DEVICE}, {N_PER_CLASS} samples/class, n_steps {N_STEPS}")

    df = pd.read_parquet(DATA / "serialized" / "test.parquet").reset_index(drop=True)
    y_true = pd.Categorical(df["Category"], categories=CLASSES).codes

    # --- 1) predict everything (batched) to find correctly-classified rows
    preds = np.empty(len(df), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(df), 128):
            chunk = df["text"].iloc[i:i + 128].tolist()
            enc = tokenizer(chunk, truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits
            preds[i:i + 128] = logits.argmax(1).cpu().numpy()
    correct = preds == y_true
    print(f"overall test acc {correct.mean():.4f}")

    # --- 2) Layer IG on the embedding layer, attributed toward the true class
    def forward_fn(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    lig = LayerIntegratedGradients(forward_fn, model.distilbert.embeddings)

    records = []          # per-sample normalized feature attribution rows
    for c, cls in enumerate(CLASSES):
        pool = np.where((y_true == c) & correct)[0]
        take = pool if len(pool) <= N_PER_CLASS else rng.choice(pool, N_PER_CLASS, replace=False)
        print(f"  {cls}: {len(take)} samples (pool {len(pool)})")
        for j, ridx in enumerate(take):
            text = df["text"].iloc[ridx]
            enc = tokenizer(text, truncation=True, max_length=MAX_LEN,
                            return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].tolist()
            input_ids = enc["input_ids"].to(DEVICE)
            attn = enc["attention_mask"].to(DEVICE)

            # reference: content tokens -> [PAD]; keep [CLS] and [SEP]
            ref = input_ids.clone()
            keep = (input_ids == cls_id) | (input_ids == sep_id)
            ref[~keep] = pad_id

            atts = lig.attribute(inputs=input_ids, baselines=ref,
                                 additional_forward_args=(attn,), target=int(c),
                                 n_steps=N_STEPS, internal_batch_size=N_STEPS)
            tok_att = atts.sum(dim=-1).squeeze(0).detach().cpu().numpy()  # per-token scalar

            spans = feature_spans(text)
            fmap = token_to_feature(offsets, spans)
            names = [s[0] for s in spans]
            feat_att = {n: 0.0 for n in names}
            for t, f in enumerate(fmap):
                if f >= 0:
                    feat_att[names[f]] += float(tok_att[t])

            total = sum(abs(v) for v in feat_att.values()) or 1.0
            row = {"class": cls, "row": int(ridx)}
            for n, v in feat_att.items():
                row[n] = v / total          # L1-normalized signed contribution
            records.append(row)
            if (j + 1) % 40 == 0:
                print(f"    {cls} {j + 1}/{len(take)}", flush=True)

    per_sample = pd.DataFrame(records).fillna(0.0)
    per_sample.to_parquet(DATA / "captum_per_sample.parquet", index=False)

    # --- 3) aggregate per class
    feat_cols = [c for c in per_sample.columns if c not in ("class", "row")]
    agg_rows = []
    for cls in CLASSES:
        sub = per_sample[per_sample["class"] == cls]
        signed = sub[feat_cols].mean()
        absmean = sub[feat_cols].abs().mean()
        order = absmean.sort_values(ascending=False)
        for rank, feat in enumerate(order.index, 1):
            agg_rows.append({"class": cls, "feature": feat,
                             "signed_mean": round(float(signed[feat]), 6),
                             "abs_mean": round(float(absmean[feat]), 6),
                             "rank": rank})
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(DATA / "captum_feature_attribution.csv", index=False)

    # --- 4) human-readable top-8 per class
    lines = ["# DistilBERT — Layer Integrated Gradients (top features per class)", "",
             f"Captum LayerIG on the embedding layer; {N_PER_CLASS} correctly-classified "
             f"test samples/class; n_steps {N_STEPS}; attributions remapped from wordpiece "
             "tokens to the 39 flow features via char offsets, L1-normalized per sample. "
             "signed_mean>0 pushes toward the class, <0 away.", ""]
    for cls in CLASSES:
        lines.append(f"## {cls}")
        top = agg[agg["class"] == cls].nsmallest(8, "rank")
        for _, r in top.iterrows():
            arrow = "+" if r["signed_mean"] >= 0 else "-"
            lines.append(f"- `{r['feature']}`  |attr|={r['abs_mean']:.4f}  ({arrow}{abs(r['signed_mean']):.4f})")
        lines.append("")
    (DATA / "captum_top_features.md").write_text("\n".join(lines))
    print("wrote captum_feature_attribution.csv, captum_per_sample.parquet, captum_top_features.md")


if __name__ == "__main__":
    main()
