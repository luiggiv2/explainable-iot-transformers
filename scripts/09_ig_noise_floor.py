"""Phase 5 diagnostic — IG noise floor (eps=0 control) for the RQ4 metric.

The perturbation study showed a flat Spearman (~0.47) across noise levels, which
is suspicious: input-driven instability should worsen with noise. This control
attributes each UNPERTURBED input twice and measures agreement, isolating how
much apparent instability comes from IG itself / the near-zero feature tail
rather than the perturbation. It also reports a top-8-restricted Spearman
(meaningful features only) as a fairer explanation-stability metric.

Prints only (no artifacts): a diagnostic to decide how to report RQ4.
"""

import importlib.util
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
MAX_LEN, N_STEPS, N_PER_CLASS = 224, 50, 15
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_spec = importlib.util.spec_from_file_location("ser", ROOT / "scripts" / "02_serialize.py")
ser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ser)
COLS = [c for c, _ in ser.FEATURES]
NAMES = [n for _, n in ser.FEATURES]


def main():
    rng = np.random.default_rng(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    pad_id, cls_id, sep_id = tok.pad_token_id, tok.cls_token_id, tok.sep_token_id
    lig = LayerIntegratedGradients(
        lambda ids, am: model(input_ids=ids, attention_mask=am).logits,
        model.distilbert.embeddings)

    df = pd.read_parquet(DATA / "splits" / "test.parquet").reset_index(drop=True)
    Xnum = df[COLS].to_numpy(dtype=np.float64)
    y = pd.Categorical(df["Category"], categories=CLASSES).codes

    def serialize_row(v):
        return " ".join(f"{n}={ser.fmt(x)}" for n, x in zip(NAMES, v))

    def attribute(text, target):
        enc = tok(text, truncation=True, max_length=MAX_LEN,
                  return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        ids, am = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
        ref = ids.clone()
        ref[~((ids == cls_id) | (ids == sep_id))] = pad_id
        att = lig.attribute(inputs=ids, baselines=ref, additional_forward_args=(am,),
                            target=int(target), n_steps=N_STEPS, internal_batch_size=25)
        ta = att.sum(-1).squeeze(0).detach().cpu().numpy()
        spans, pos = [], 0
        for n in NAMES:
            s = text.index(f"{n}=", pos)
            e = text.index(" ", s) if text.find(" ", s) != -1 else len(text)
            spans.append((s, e)); pos = e
        vec = np.zeros(len(NAMES))
        for t, (s, e) in enumerate(offs):
            if s == e:
                continue
            for f, (fs, fe) in enumerate(spans):
                if fs <= s < fe:
                    vec[f] += float(ta[t]); break
        return np.abs(vec)

    # predict to get correctly-classified pool
    preds = np.empty(len(df), np.int64)
    txt = [serialize_row(Xnum[i]) for i in range(len(df))]
    for i in range(0, len(df), 256):
        enc = tok(txt[i:i+256], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            preds[i:i+256] = model(**enc).logits.argmax(1).cpu().numpy()
    correct = preds == y

    full, top8 = [], []
    for c, cls in enumerate(CLASSES):
        pool = np.where((y == c) & correct)[0]
        take = rng.choice(pool, min(N_PER_CLASS, len(pool)), replace=False)
        fc, t8 = [], []
        for ridx in take:
            t = serialize_row(Xnum[ridx])
            a1, a2 = attribute(t, c), attribute(t, c)         # same input, twice
            fc.append(spearmanr(a1, a2).correlation)
            idx = np.argsort(a1)[::-1][:8]                     # top-8 by first run
            r = spearmanr(a1[idx], a2[idx]).correlation
            t8.append(r if r == r else 1.0)
        full += fc; top8 += t8
        print(f"  {cls:12s} eps0 Spearman(all39)={np.nanmean(fc):.3f}  top8={np.nanmean(t8):.3f}")
    print(f"\nNOISE FLOOR (eps=0, identical input attributed twice):")
    print(f"  Spearman over all 39 features: {np.nanmean(full):.3f}")
    print(f"  Spearman over top-8 features : {np.nanmean(top8):.3f}")


if __name__ == "__main__":
    main()
