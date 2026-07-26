"""Phase 1 — serialize CICIoT2023 flow windows to template text.

Each row becomes a fixed-order "name=value" string over the 39 window-aggregated
features of the regenerated CSV schema (this version exposes protocol-indicator
fractions rather than raw ports — the serialization keeps a 1:1 name mapping to
the original columns so Captum/SHAP attributions can be mapped back).

Floats are trimmed to 4 significant digits to bound token noise; integers stay
integers. Feature order is domain-grouped (protocol context -> TCP flags ->
app-protocol indicators -> size stats -> timing) and identical for every row.

Outputs: data/serialized/{train,val,test}.parquet  (text, Label, Category)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "data" / "serialized"

# original column -> serialized token name (1:1, fixed order)
FEATURES = [
    # protocol / transport context
    ("Protocol Type", "proto_num"),
    ("Time_To_Live", "ttl"),
    ("Rate", "rate"),
    ("Header_Length", "header_len"),
    # TCP flag fractions within window
    ("fin_flag_number", "fin"),
    ("syn_flag_number", "syn"),
    ("rst_flag_number", "rst"),
    ("psh_flag_number", "psh"),
    ("ack_flag_number", "ack"),
    ("ece_flag_number", "ece"),
    ("cwr_flag_number", "cwr"),
    # flag counts
    ("ack_count", "ack_cnt"),
    ("syn_count", "syn_cnt"),
    ("fin_count", "fin_cnt"),
    ("rst_count", "rst_cnt"),
    # application/link protocol indicator fractions
    ("HTTP", "http"),
    ("HTTPS", "https"),
    ("DNS", "dns"),
    ("Telnet", "telnet"),
    ("SMTP", "smtp"),
    ("SSH", "ssh"),
    ("IRC", "irc"),
    ("TCP", "tcp"),
    ("UDP", "udp"),
    ("DHCP", "dhcp"),
    ("ARP", "arp"),
    ("ICMP", "icmp"),
    ("IGMP", "igmp"),
    ("IPv", "ipv"),
    ("LLC", "llc"),
    # packet-size statistics over window
    ("Tot sum", "tot_sum"),
    ("Min", "min"),
    ("Max", "max"),
    ("AVG", "avg"),
    ("Std", "std"),
    ("Tot size", "tot_size"),
    # timing / volume
    ("IAT", "iat"),
    ("Number", "num"),
    ("Variance", "var"),
]


def fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    f = float(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.4g}"


def serialize(df: pd.DataFrame) -> pd.Series:
    cols = [c for c, _ in FEATURES]
    names = [n for _, n in FEATURES]
    values = df[cols].to_numpy()
    return pd.Series(
        [" ".join(f"{n}={fmt(v)}" for n, v in zip(names, row)) for row in values],
        index=df.index, name="text",
    )


def main():
    OUT.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        df = pd.read_parquet(SPLITS / f"{split}.parquet")
        out = pd.DataFrame({
            "text": serialize(df),
            "Label": df["Label"],
            "Category": df["Category"],
        })
        out.to_parquet(OUT / f"{split}.parquet", index=False)
        print(f"{split}: {len(out)} rows serialized")

    # show one example per selected category + token-length stats
    test = pd.read_parquet(OUT / "test.parquet")
    for cat in ("Benign", "Mirai", "DDoS", "Recon"):
        row = test[test["Category"] == cat].iloc[0]
        print(f"\n[{cat} / {row['Label']}]\n{row['text']}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    sample = test["text"].sample(2000, random_state=42)
    lens = [len(tok(t)["input_ids"]) for t in sample]
    print(f"\nToken lengths (n=2000): mean={np.mean(lens):.0f} "
          f"p95={np.percentile(lens, 95):.0f} max={max(lens)} (model limit 512)")


if __name__ == "__main__":
    main()
