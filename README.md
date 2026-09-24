# Explainable Lightweight Transformers for Malicious IoT Traffic Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21584810.svg)](https://doi.org/10.5281/zenodo.21584810)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducibility package for the paper *"Explainable Lightweight Transformers for
Malicious IoT Traffic Detection"* (working title). This repository contains the
full, script-driven pipeline — data preparation, classical baselines, DistilBERT
fine-tuning, dual-method explainability (Captum Integrated Gradients + SHAP),
explanation-robustness analysis, statistical tests, and figure generation — on
the **CICIoT2023** dataset.

> **Author:** Luiggi Valencia Vélez · **Status:** manuscript in preparation for submission · **Target:** *Internet of Things* (Elsevier), Subscription route · **ORCID:** [0009-0003-7783-9023](https://orcid.org/0009-0003-7783-9023)

> **Version note:** this version supersedes v1.0.0. An audit of the first
> experiment found cross-split serialization duplicates and a mismatch between a
> separately exported FP32 prediction array and the BF16-reported confusion
> matrix. All results are now leakage-safe, run-owned artifacts under
> `data/revision/`; the superseded pipeline remains available in the v1.0.0
> archive on Zenodo.

---

## Overview

We fine-tune a pretrained **DistilBERT** encoder as an 8-class sequence
classifier over a **textual `key=value` serialization** of CICIoT2023 flow-window
features, and compare it against Random Forest, SVM, and XGBoost baselines under
a strict per-class protocol. The contribution is **not** accuracy or efficiency —
the encoder is competitive but not superior to gradient-boosted trees, and is the
heaviest model — but an **auditing-oriented explainability analysis**:

- **RQ1** — DistilBERT reaches macro-F1 **0.7058 ± 0.0112** over three seeds
  (XGBoost 0.7278, RF 0.7223, SVM 0.6672). The deficit against XGBoost holds in
  every seed; the deficit against RF is **unresolved** — its bootstrap interval
  contains zero in two seeds of three and McNemar never separates the two.
- **RQ2** — Integrated Gradients (on DistilBERT) and SHAP (on XGBoost) reach
  comparable accuracy from **measurably different evidence**: the tree ranks `ssh`
  first for BruteForce, the encoder ranks it 7th of 39. Mean rank correlation
  +0.464, top-5 overlap 37.5%. A gradient-free masking check confirms the encoder's
  profile is real, not an attribution artifact.
- **RQ3** — Attribution exposes dependencies invisible to aggregate metrics: a
  spurious ARP dependency (Spoofing — 88.2% of those windows carry no ARP traffic),
  divergent DDoS strategies, and a **window-size collection artifact** both model
  families exploit. Retraining without it costs only 0.0057 macro-F1, so the
  shortcut is redundant rather than load-bearing — and removing it restores `ssh`
  as the encoder's brute-force signal.
- **RQ4** — Explanation stability co-varies with prediction stability across classes
  (r = **+0.959**, 95% CI [+0.897, +0.977]), and survives dropping the two
  volumetric classes (r = +0.982, p = 0.0069). An ε = 0 control is **bitwise
  identical**, so the measured instability is input sensitivity, not method noise.
- **RQ5** — Under a **controlled same-CPU benchmark** (4 threads pinned), the
  transformer is the heaviest option: 175× XGBoost's footprint, 123× its latency at
  batch 1 and 2,139× at batch 64 — batching helps the trees 19–31× and the encoder
  1.08×, so the gap widens.

Every quantitative claim in the paper traces to a manifest-backed artifact under
`data/revision/`.

---

## Repository structure

```
scripts/            Numbered, seeded pipeline (run in order; see below)
data/                Raw-class scan used by script 01 (class_distribution_raw.csv).
                     Raw/derived dataset and models are NOT committed —
                     regenerate them (see "Data" and "Reproduce").
  revision/          Result artifacts: metrics, predictions, confusion matrices,
                     XAI outputs, statistics and figures, each with a manifest
requirements-lock.txt  Pinned dependency versions
```

Not tracked (see `.gitignore`): `data/raw/` (the CICIoT2023 CSVs), `data/models/`
and the derived `data/{splits,serialized,subset_60k.parquet}` (regenerable),
and `.venv/`.

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
> for CPU-only stats — duplicate OpenMP runtimes crash the interpreter. Script
> `29` isolates each model family in its own child process for this reason. In long Captum loops, call
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
Each writes its outputs under `data/revision/`.

| Step | Script | Produces |
|---|---|---|
| 1 | `01_make_subset.py` | Stratified ~60k subset + leakage-safe grouped splits + source IDs |
| 2 | `02_serialize.py` | Registered original / feature-ablation serializations |
| 3 | `03_baselines.py` | Run-owned RF / SVM / XGBoost metrics, predictions and models |
| 4 | `04_finetune.py` | Multiseed DistilBERT runs with FP32 evaluation and checksums |
| 17 | `17_reproducibility_audit.py` | Split, duplicate, hash and prediction-consistency audit |
| 18 | `18_feature_ablation_summary.py` | Registered tree ablations and shortcut diagnostics |
| 19 | `19_compare_predictions.py` | `SampleID`-paired model tests and macro-F1 bootstrap |
| 20 | `20_multiseed_summary.py` | Across-seed aggregation per feature set |
| 21 | `21_compare_feature_sets.py` | Paired original-vs-`no_number` comparison |
| 22 | `22_make_xai_cohorts.py` | Fixed, hashed XAI cohorts (jointly correct rows) |
| 23 | `23_captum_ig_revised.py` | Layer-IG attributions + convergence audit |
| 24 | `24_shap_xgb_revised.py` | TreeSHAP on the identical cohort |
| 25 | `25_xai_synthesis_revised.py` | Cross-model IG-vs-SHAP comparison (RQ2/RQ3) |
| 26 | `26_compare_ig_feature_sets.py` | IG before/after removing `Number` |
| 27 | `27_ablation_crosscheck_revised.py` | Gradient-free field-masking cross-check |
| 28 | `28_robustness_revised.py` | Perturbation-based explanation robustness (RQ4) |
| 29 | `29_cpu_inference_benchmark.py` | Controlled same-CPU latency benchmark (RQ5) |
| 30 | `30_ig_quality_audit.py` | Relative IG convergence-residual audit |
| 31 | `31_ig_noise_floor_revised.py` | ε = 0 attribution floor on the robustness cohort |
| 33 | `33_attribution_rank_stability.py` | Bootstrap stability of per-class rankings |
| 34 | `34_robustness_coupling.py` | Class-level prediction/explanation stability coupling (r, flow-cluster bootstrap CI) |

These scripts produce every number in the current manuscript. The superseded
single-seed pipeline (former scripts 05–16) and its outputs are preserved in the
v1.0.0 archive on Zenodo and are not part of this version.

Steps 22–34 depend on the models from steps 3–4. The revised DistilBERT protocol
uses seeds 42, 123 and 2026 for the two principal configurations; diagnostic
ablations begin with seed 42 and expand only if the registered effect threshold
is met.

---

---

## Citing

If you use this code, please cite the paper (citation to be added on publication),
this software archive, and the CICIoT2023 dataset:

```bibtex
@software{valenciavelez2026explainableiot,
  author    = {Valencia V{\'e}lez, Luiggi},
  title     = {Explainable Lightweight Transformers for Malicious IoT
               Traffic Detection},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.1.0},
  doi       = {10.5281/zenodo.21584810},
  url       = {https://doi.org/10.5281/zenodo.21584810}
}

@inproceedings{neto2023ciciot,
  title     = {CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale
               Attacks in IoT Environment},
  author    = {Neto, Euclides Carlos Pinto and others},
  year      = {2023}
}
```

---

## License

Code and derived artifacts in this repository are released under the **MIT
License** (see [`LICENSE`](LICENSE)). The CICIoT2023 dataset is **not** included
and remains under the CIC's terms of use.

## Notes

Portions of the code, drafting, and analysis were produced with AI-assisted
tooling under author supervision; all experimental design, results, and
scientific claims were verified and approved by the authors.
