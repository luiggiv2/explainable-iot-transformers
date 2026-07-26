"""Phase 5 — explanation robustness under minor input perturbations (RQ4).

Take correctly-classified test flows, apply small semantics-preserving noise to
their numeric features, re-serialize with the training-identical template, re-run
Layer Integrated Gradients toward the same class, and measure how stable the
feature-importance ranking is (Spearman of the 39-feature attribution vectors,
plus top-5 overlap). An uncommon methodological contribution: a model whose
explanations flip under trivial noise cannot be trusted for deployment.

Perturbation: multiplicative Gaussian x' = x*(1+eps*N(0,1)) at eps in {1%,5%,10%}.
proto_num (protocol identity) is held fixed; [0,1] fraction features are clamped
to [0,1]; magnitudes clamped to >=0. Structure (which protocols are present) is
preserved — only magnitudes jitter, as measurement noise would.

Outputs (data/):
  robustness_metrics.csv    per class x eps: mean Spearman, top5 Jaccard, pred-preserved frac, n
  robustness_summary.md     human-readable robustness curve
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = DATA / "models" / "distilbert"
CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
MAX_LEN = 224
N_STEPS = 50
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", "60"))
EPS_LEVELS = [0.01, 0.05, 0.10]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# reuse the exact training-time serialization (FEATURES order + number format)
_spec = importlib.util.spec_from_file_location("ser", ROOT / "scripts" / "02_serialize.py")
ser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ser)
COLS = [c for c, _ in ser.FEATURES]
NAMES = [n for _, n in ser.FEATURES]
FRACTION = {"fin", "syn", "rst", "psh", "ack", "ece", "cwr", "http", "https",
            "dns", "telnet", "smtp", "ssh", "irc", "tcp", "udp", "dhcp", "arp",
            "icmp", "igmp", "ipv", "llc"}
FIXED = {"proto_num"}                     # protocol identity — never perturbed
frac_idx = np.array([i for i, n in enumerate(NAMES) if n in FRACTION])
fixed_idx = np.array([i for i, n in enumerate(NAMES) if n in FIXED])


def serialize_row(vec):
    return " ".join(f"{n}={ser.fmt(v)}" for n, v in zip(NAMES, vec))


def perturb(vec, eps, rng):
    x = vec.astype(np.float64).copy()
    noise = 1.0 + eps * rng.standard_normal(len(x))
    x = x * noise
    x[fixed_idx] = vec[fixed_idx]                 # keep protocol id exactly
    x = np.maximum(x, 0.0)                         # magnitudes stay >= 0
    x[frac_idx] = np.clip(x[frac_idx], 0.0, 1.0)   # fractions stay in [0,1]
    return x


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    pad_id, cls_id, sep_id = tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id
    print(f"device {DEVICE}, {N_PER_CLASS}/class, eps {EPS_LEVELS}, n_steps {N_STEPS}")

    df = pd.read_parquet(DATA / "splits" / "test.parquet").reset_index(drop=True)
    Xnum = df[COLS].to_numpy(dtype=np.float64)
    y = pd.Categorical(df["Category"], categories=CLASSES).codes

    def predict_text(texts):
        enc = tokenizer(texts, truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            return model(**enc).logits.argmax(1).cpu().numpy()

    # correctly-classified pool per class (predict all, batched)
    preds = np.empty(len(df), dtype=np.int64)
    texts_all = [serialize_row(Xnum[i]) for i in range(len(df))]
    for i in range(0, len(df), 256):
        preds[i:i + 256] = predict_text(texts_all[i:i + 256])
    correct = preds == y
    print(f"test acc {correct.mean():.4f}")

    def forward_fn(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
    lig = LayerIntegratedGradients(forward_fn, model.distilbert.embeddings)

    def attribute(text, target):
        """Return (abs 39-vector in NAMES order, top predicted class)."""
        enc = tokenizer(text, truncation=True, max_length=MAX_LEN,
                        return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"].to(DEVICE)
        attn = enc["attention_mask"].to(DEVICE)
        ref = ids.clone()
        keep = (ids == cls_id) | (ids == sep_id)
        ref[~keep] = pad_id
        att = lig.attribute(inputs=ids, baselines=ref, additional_forward_args=(attn,),
                            target=int(target), n_steps=N_STEPS, internal_batch_size=25)
        tok_att = att.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        # map wordpiece -> feature via char spans (fields are space-separated key=value)
        spans, pos = [], 0
        for n in NAMES:
            start = text.index(f"{n}=", pos)
            end = text.index(" ", start) if text.find(" ", start) != -1 else len(text)
            spans.append((start, end)); pos = end
        vec = np.zeros(len(NAMES))
        for t, (s, e) in enumerate(offsets):
            if s == e:
                continue
            for f, (fs, fe) in enumerate(spans):
                if fs <= s < fe:
                    vec[f] += float(tok_att[t]); break
        return np.abs(vec)

    def top5(vec):
        return set(np.argsort(vec)[::-1][:5])

    def free_mps():
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    records = []
    for c, cls in enumerate(CLASSES):
        pool = np.where((y == c) & correct)[0]
        take = pool if len(pool) <= N_PER_CLASS else rng.choice(pool, N_PER_CLASS, replace=False)
        for j, ridx in enumerate(take):
            x0 = Xnum[ridx]
            a0 = attribute(serialize_row(x0), c)
            base_top = top5(a0)
            for eps in EPS_LEVELS:
                xp = perturb(x0, eps, rng)
                tp = serialize_row(xp)
                ap = attribute(tp, c)
                pred_p = int(predict_text([tp])[0])
                rho = spearmanr(a0, ap).correlation
                jac = len(base_top & top5(ap)) / len(base_top | top5(ap))
                records.append({"class": cls, "eps": eps,
                                "spearman": float(rho) if rho == rho else np.nan,
                                "jaccard5": jac, "pred_preserved": pred_p == c})
            if (j + 1) % 10 == 0:
                free_mps()                        # release MPS cache to avoid slowdown
            if (j + 1) % 20 == 0:
                print(f"    {cls} {j + 1}/{len(take)}", flush=True)
        # per-class checkpoint: never lose completed work again
        pd.DataFrame(records).to_csv(DATA / "robustness_records_partial.csv", index=False)
        print(f"  {cls}: {len(take)} samples done (checkpoint saved)", flush=True)

    rec = pd.DataFrame(records)
    # Explanation stability = Spearman/Jaccard of the class-c attribution map over ALL
    # perturbed samples (well-defined regardless of the argmax). Prediction stability
    # is reported separately as pred_preserved_frac.
    rows = []
    for cls in CLASSES + ["_ALL"]:
        for eps in EPS_LEVELS:
            sub = rec[rec["eps"] == eps] if cls == "_ALL" else rec[(rec["class"] == cls) & (rec["eps"] == eps)]
            rows.append({
                "class": cls, "eps": eps, "n": len(sub),
                "pred_preserved_frac": round(sub["pred_preserved"].mean(), 4),
                "mean_spearman": round(sub["spearman"].mean(), 4),
                "mean_jaccard5": round(sub["jaccard5"].mean(), 4),
            })
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "robustness_metrics.csv", index=False)

    L = ["# Explanation robustness under perturbation (RQ4)", "",
         f"DistilBERT Layer-IG; {N_PER_CLASS} correctly-classified test flows/class; "
         f"multiplicative Gaussian noise eps in {EPS_LEVELS} (proto_num fixed, fractions "
         "clamped to [0,1]); re-serialized identically and re-attributed toward the true "
         "class. Spearman/top-5 Jaccard measure stability of the class attribution map "
         "over all perturbed samples; prediction stability is reported separately.", "",
         "## Overall robustness curve (all classes)", "",
         "| eps | pred preserved | mean Spearman | mean top-5 Jaccard |",
         "|---|---|---|---|"]
    for _, r in out[out["class"] == "_ALL"].iterrows():
        L.append(f"| {r['eps']:.0%} | {r['pred_preserved_frac']:.1%} | "
                 f"{r['mean_spearman']:.3f} | {r['mean_jaccard5']:.3f} |")
    L += ["", "## Per-class mean Spearman by eps", "",
          "| Class | " + " | ".join(f"{e:.0%}" for e in EPS_LEVELS) + " |",
          "|---|" + "---|" * len(EPS_LEVELS)]
    for cls in CLASSES:
        vals = [out[(out["class"] == cls) & (out["eps"] == e)]["mean_spearman"].iloc[0] for e in EPS_LEVELS]
        L.append(f"| {cls} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    L += ["", "## Per-class prediction preserved by eps", "",
          "| Class | " + " | ".join(f"{e:.0%}" for e in EPS_LEVELS) + " |",
          "|---|" + "---|" * len(EPS_LEVELS)]
    for cls in CLASSES:
        vals = [out[(out["class"] == cls) & (out["eps"] == e)]["pred_preserved_frac"].iloc[0] for e in EPS_LEVELS]
        L.append(f"| {cls} | " + " | ".join(f"{v:.1%}" for v in vals) + " |")
    L += [""]
    (DATA / "robustness_summary.md").write_text("\n".join(L))
    print("wrote data/robustness_metrics.csv, data/robustness_summary.md")


if __name__ == "__main__":
    main()
