# Explainable Lightweight Transformers for Malicious IoT Traffic Detection

Reproducibility package for the paper *"Explainable Lightweight Transformers for
Malicious IoT Traffic Detection"* (working title). This repository contains the
full, script-driven pipeline — data preparation, classical baselines, DistilBERT
fine-tuning, dual-method explainability (Captum Integrated Gradients + SHAP),
explanation-robustness analysis, statistical tests, and figure generation — on
the **CICIoT2023** dataset.

> **Authors:** _[fill in]_ · **Status:** under review · **Contact:** _[fill in]_

---

## Overview

We fine-tune a pretrained **DistilBERT** encoder as an 8-class sequence
classifier over a **textual `key=value` serialization** of CICIoT2023 flow-window
features, and compare it against Random Forest, SVM, and XGBoost baselines under
a strict per-class protocol. The contribution is **not** accuracy or efficiency —
the encoder is competitive but not superior to gradient-boosted trees, and is the
heaviest model — but an **auditing-oriented explainability analysis**:

- **RQ1** — DistilBERT reaches macro-F1 **0.707** (XGBoost 0.734, RF 0.726, SVM
  0.655); below XGBoost on macro-F1, indistinguishable from RF on accuracy.
- **RQ2** — Integrated Gradients (on DistilBERT) and SHAP (on XGBoost)
  independently recover documented attack signatures (`ssh`→BruteForce, size
  statistics→Mirai, `syn`→Recon), corroborated by a gradient-free ablation check.
- **RQ3** — Attribution exposes systematic dependencies invisible to aggregate
  metrics: a spurious ARP dependency (Spoofing), divergent DDoS strategies, and a
  **window-size collection artifact** both models exploit.
- **RQ4** — Explanation stability co-varies with prediction stability across
  classes (r ≈ 0.85), governed by class separability; a predicted-class control
  shows the coupling is empirical, not definitional.
- **RQ5** — The transformer is the heaviest option (~172× XGBoost's footprint,
  ~6,580× its per-sample latency), with no offsetting accuracy advantage.

Every quantitative claim in the paper traces to a named artifact under `data/`.

---

## Repository structure

```
scripts/            Numbered, seeded pipeline (run in order; see below)
data/                Result artifacts (metrics, confusion matrices, findings,
                     stats, figures). Raw/derived dataset and models are NOT
                     committed — regenerate them (see "Data" and "Reproduce").
  figures/           Publication figures (PDF + PNG) and their index
references/          refs.bib (verified) + DOI verification logs
research/            One Markdown fact sheet per reviewed paper (literature notes)
requirements-lock.txt  Pinned dependency versions
```

Not tracked (see `.gitignore`): `data/raw/` (the CICIoT2023 CSVs), `data/models/`
and the derived `data/{splits,serialized,subset_60k.parquet}` (regenerable),
`.venv/`, and the copyrighted publisher PDFs.

---

## Requirements

- **Python 3.12** (the reported results used CPython **3.12.8**)
- A machine with the Apple **MPS** backend (the paper's training used an Apple M3);
  the code also runs on CPU/CUDA with minor device changes.
- Dependencies are pinned in [`requirements-lock.txt`](requirements-lock.txt)
  (key packages: `torch` 2.13.0, `transformers` 5.14.0, `scikit-learn` 1.9.0,
  `xgboost` 3.3.0, `captum` 0.9.0, `shap` 0.52.0).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

> **macOS note:** do not `import torch` in the same process as XGBoost/scikit-learn
> for CPU-only stats — duplicate OpenMP runtimes crash the interpreter. Scripts
> `13`/`15`/`16` are deliberately torch-free. In long Captum loops, call
> `torch.mps.empty_cache()` periodically (already done in the scripts) to avoid
> MPS slowdown.

---

## Data

This study uses **CICIoT2023**, distributed by the Canadian Institute for
Cybersecurity (University of New Brunswick). **We do not redistribute it.**
Download the 39-feature "MergedCSV" release from the CIC and place the 63
`MergedXX.csv` files under `data/raw/`:

- CICIoT2023: <https://www.unb.ca/cic/datasets/iotdataset-2023.html>

Please cite the dataset paper (Neto et al., 2023) and comply with the CIC terms
of use. All derived data (the stratified subset, splits, and serialized text) is
regenerated from these files by scripts `01`–`02`.

---

## Reproduce

Run the numbered scripts in order (from the repo root, with the venv active).
Each writes its outputs under `data/`.

| Step | Script | Produces |
|---|---|---|
| 1 | `01_make_subset.py` | Stratified ~60k subset + 70/15/15 splits (seed 42) |
| 2 | `02_serialize.py` | `key=value` textual serialization of each flow window |
| 3 | `03_baselines.py` | RF / SVM / XGBoost metrics, confusion matrices, models |
| 4 | `04_finetune.py` | DistilBERT fine-tuning; metrics, training log, checkpoint |
| 5 | `05_captum_ig.py` | Layer Integrated Gradients attributions (per class) |
| 6 | `06_shap_xgb.py` | TreeSHAP attributions for XGBoost (same protocol) |
| 7 | `07_xai_synthesis.py` | Cross-model IG-vs-SHAP comparison (RQ2/RQ3) |
| 8 | `08_robustness.py` | Perturbation-based explanation robustness (RQ4) |
| 9 | `09_ig_noise_floor.py` | ε=0 IG determinism control |
| 10 | `10_robustness_synthesis.py` | RQ4 findings synthesis |
| 11 | `11_figures.py` | Publication figures → `data/figures/` |
| 12 | `12_ablation_crosscheck.py` | Gradient-free ablation cross-check of IG |
| 13 | `13_stats.py` | McNemar, RQ4 coupling CI/permutation, ranking bootstrap |
| 14 | `14_predicted_class_control.py` | RQ4 predicted-class control |
| 15 | `15_macrof1_boot.py` | Macro-F1 difference bootstrap + RF seed variance |
| 16 | `16_dataset_stats.py` | Per-class `Number`/`ARP` provenance stats |

Steps 5–16 depend on the models from steps 3–4. All randomness uses **seed 42**.

---

## Key results (test split, n = 9,000)

| Model | Macro-F1 | Accuracy | ROC-AUC | Size | Latency |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | **0.7344** | 0.7993 | 0.9743 | 1.6 MB | 0.0016 ms |
| Random Forest | 0.7258 | 0.7849 | 0.9695 | 60 MB | 0.0086 ms |
| **DistilBERT** | **0.7069** | 0.7811 | 0.9706 | 269 MB | 10.37 ms |
| SVM | 0.6554 | 0.7694 | 0.9629 | 1.0 MB | 0.81 ms |

Full breakdowns: `data/baselines_metrics.csv`, `data/transformer_metrics.csv`,
`data/stats_hardening.md`, `data/macrof1_boot.md`, `data/xai_comparison.md`,
`data/robustness_findings.md`, `data/predclass_control.md`, `data/figures/`.

---

## Citing

If you use this code, please cite the paper (citation to be added on publication)
and the CICIoT2023 dataset:

```bibtex
@inproceedings{neto2023ciciot,
  title     = {CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale
               Attacks in IoT Environment},
  author    = {Neto, Euclides Carlos Pinto and others},
  year      = {2023}
}
```

(See `references/refs.bib` for the full, DOI-verified bibliography.)

---

## License

Code and derived artifacts in this repository are released under the **MIT
License** (see [`LICENSE`](LICENSE)). The CICIoT2023 dataset is **not** included
and remains under the CIC's terms of use.

## Notes

Portions of the code, drafting, and analysis were produced with AI-assisted
tooling under author supervision; all experimental design, results, and
scientific claims were verified and approved by the authors.
