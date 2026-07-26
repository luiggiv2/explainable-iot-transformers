"""Phase B (review) — RQ4 predicted-class control (reviewer issue #3b).

The robustness study attributes toward the true class even after a perturbed
prediction has flipped, so some top-5 reshuffling is definitional. Control:
attribute each input toward its OWN predicted class (original -> pred p0,
perturbed -> pred p1), so top-5 Jaccard measures stability of the explanation
of the *current decision*. If the prediction<->explanation coupling weakens
under this control, part of the r=+0.85 was definitional; if it holds, the
coupling is empirical.

Imports NO xgboost/sklearn (torch + those crash via duplicate OpenMP on macOS).
Output: data/predclass_control.md
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from scipy.stats import pearsonr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = DATA / "models" / "distilbert"
CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
MAX_LEN, N_STEPS, N_PER_CLASS = 224, 50, 60
EPS_LEVELS = [0.01, 0.05, 0.10]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
rng = np.random.default_rng(SEED)

_spec = importlib.util.spec_from_file_location("ser", ROOT / "scripts" / "02_serialize.py")
ser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ser)
COLS = [c for c, _ in ser.FEATURES]
NAMES = [n for _, n in ser.FEATURES]
FRACTION = {"fin", "syn", "rst", "psh", "ack", "ece", "cwr", "http", "https", "dns",
            "telnet", "smtp", "ssh", "irc", "tcp", "udp", "dhcp", "arp", "icmp",
            "igmp", "ipv", "llc"}
frac_idx = np.array([i for i, n in enumerate(NAMES) if n in FRACTION])
fixed_idx = np.array([i for i, n in enumerate(NAMES) if n == "proto_num"])


def serialize_row(v):
    return " ".join(f"{n}={ser.fmt(x)}" for n, x in zip(NAMES, v))


def perturb(vec, eps):
    x = vec.astype(np.float64) * (1.0 + eps * rng.standard_normal(len(vec)))
    x[fixed_idx] = vec[fixed_idx]
    x = np.maximum(x, 0.0)
    x[frac_idx] = np.clip(x[frac_idx], 0.0, 1.0)
    return x


def main():
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    pad_id, cls_id, sep_id = tok.pad_token_id, tok.cls_token_id, tok.sep_token_id
    lig = LayerIntegratedGradients(
        lambda ids, am: model(input_ids=ids, attention_mask=am).logits,
        model.distilbert.embeddings)
    print(f"device {DEVICE}, {N_PER_CLASS}/class", flush=True)

    df = pd.read_parquet(DATA / "splits" / "test.parquet").reset_index(drop=True)
    Xnum = df[COLS].to_numpy(np.float64)
    y = pd.Categorical(df["Category"], categories=CLASSES).codes

    def predict(text):
        enc = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            return int(model(**enc).logits.argmax(1).item())

    def top5_toward(text, target):
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
        return set(np.argsort(np.abs(vec))[::-1][:5])

    # correctly-classified pool (batched prediction over serialized text)
    texts_all = [serialize_row(Xnum[i]) for i in range(len(df))]
    preds0 = np.empty(len(df), np.int64)
    with torch.no_grad():
        for i in range(0, len(df), 128):
            enc = tok(texts_all[i:i+128], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(DEVICE)
            preds0[i:i+128] = model(**enc).logits.argmax(1).cpu().numpy()
    correct = preds0 == y
    print(f"acc {correct.mean():.4f}", flush=True)

    rows = []
    for c, cls in enumerate(CLASSES):
        pool = np.where((y == c) & correct)[0]
        take = pool if len(pool) <= N_PER_CLASS else rng.choice(pool, N_PER_CLASS, replace=False)
        for j, ridx in enumerate(take):
            x0 = Xnum[ridx]
            base_top = top5_toward(serialize_row(x0), c)   # toward true (=pred) class
            for eps in EPS_LEVELS:
                tp = serialize_row(perturb(x0, eps))
                p1 = predict(tp)
                top_pred = top5_toward(tp, p1)              # toward the input's OWN prediction
                jac = len(base_top & top_pred) / len(base_top | top_pred)
                rows.append({"class": cls, "eps": eps, "jaccard5_predclass": jac,
                             "pred_preserved": p1 == c})
            if (j + 1) % 10 == 0 and DEVICE.type == "mps":
                torch.mps.empty_cache()                     # avoid MPS slowdown
        pd.DataFrame(rows).to_csv(DATA / "predclass_control_records.csv", index=False)
        print(f"  {cls} done ({len(take)} samples, checkpoint saved)", flush=True)

    rec = pd.DataFrame(rows)
    rec.to_csv(DATA / "predclass_control_records.csv", index=False)

    per = rec.groupby("class").agg(p=("pred_preserved", "mean"),
                                   j=("jaccard5_predclass", "mean")).reindex(CLASSES)
    r_pred = pearsonr(per["p"], per["j"])[0]
    # compare to the true-class coupling from the original robustness records
    orig = pd.read_csv(DATA / "robustness_records_partial.csv")
    po = orig.groupby("class").agg(p=("pred_preserved", "mean"),
                                   j=("jaccard5", "mean")).reindex(CLASSES)
    r_true = pearsonr(po["p"], po["j"])[0]
    kept = rec[rec["pred_preserved"]]["jaccard5_predclass"].mean()
    flip = rec[~rec["pred_preserved"]]["jaccard5_predclass"].mean()

    L = ["# RQ4 predicted-class control (reviewer issue #3b)", "",
         f"Attributing each input toward its OWN predicted class (perturbed -> its "
         f"prediction), {N_PER_CLASS} correctly-classified rows/class, eps {EPS_LEVELS}.", "",
         f"- coupling r (pred-preserved vs top-5 Jaccard) attributing toward PREDICTED "
         f"class: **{r_pred:+.3f}**",
         f"- for reference, attributing toward TRUE class (original analysis): {r_true:+.3f}",
         f"- top-5 Jaccard (pred-class) when prediction preserved: {kept:.3f}; when flipped: {flip:.3f}",
         "",
         "| Class | pred preserved | top-5 Jaccard (pred-class) |",
         "|---|---|---|"]
    for c in CLASSES:
        L.append(f"| {c} | {per.loc[c,'p']:.1%} | {per.loc[c,'j']:.3f} |")
    L += ["",
          "Reading: if the predicted-class coupling stays close to the true-class value, "
          "the r=+0.85 is largely empirical rather than definitional; a large drop would "
          "indicate the true-class attribution inflated the low-rate fragility.", ""]
    (DATA / "predclass_control.md").write_text("\n".join(L))
    print("\n".join(L), flush=True)
    print("wrote data/predclass_control.md", flush=True)


if __name__ == "__main__":
    main()
