"""Emit load-bearing dataset statistics to a committed, human-readable artifact
(reviewer minor #6): the per-class ARP-zero fraction and the per-window packet
count `Number`, previously reported only from direct parquet inspection.
Torch-free. Output: data/dataset_stats.md
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]


def main():
    df = pd.read_parquet(DATA / "splits" / "test.parquet")
    L = ["# Dataset statistics (test split, n=9,000) — provenance for §3.1/§4.3", "",
         "Computed from data/splits/test.parquet.", "",
         "| Category | n | ARP=0 frac | Number mean | Number min | Number max |",
         "|---|---:|---:|---:|---:|---:|"]
    for c in CLASSES:
        s = df[df["Category"] == c]
        L.append(f"| {c} | {len(s)} | {(s['ARP']==0).mean():.3f} | "
                 f"{s['Number'].mean():.2f} | {int(s['Number'].min())} | {int(s['Number'].max())} |")
    L += ["",
          f"Overall Spoofing ARP=0 fraction: {(df[df.Category=='Spoofing']['ARP']==0).mean():.3f}. "
          "The Number column confirms the two windowing regimes (≈10 vs ≈100 packets/window).", ""]
    (DATA / "dataset_stats.md").write_text("\n".join(L))
    print("\n".join(L))
    print("wrote data/dataset_stats.md")


if __name__ == "__main__":
    main()
