"""Shared experimental constants for the CICIoT2023 revision pipeline.

Keeping the feature order and class order in one module prevents silent row or
column drift between the tabular, serialized, prediction, and XAI stages.
"""

from __future__ import annotations

import numpy as np


SEED = 42

CLASSES = [
    "Benign",
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web",
]

# Original CICIoT2023 column -> serialized field name, in model input order.
FEATURE_PAIRS = [
    ("Protocol Type", "proto_num"),
    ("Time_To_Live", "ttl"),
    ("Rate", "rate"),
    ("Header_Length", "header_len"),
    ("fin_flag_number", "fin"),
    ("syn_flag_number", "syn"),
    ("rst_flag_number", "rst"),
    ("psh_flag_number", "psh"),
    ("ack_flag_number", "ack"),
    ("ece_flag_number", "ece"),
    ("cwr_flag_number", "cwr"),
    ("ack_count", "ack_cnt"),
    ("syn_count", "syn_cnt"),
    ("fin_count", "fin_cnt"),
    ("rst_count", "rst_cnt"),
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
    ("Tot sum", "tot_sum"),
    ("Min", "min"),
    ("Max", "max"),
    ("AVG", "avg"),
    ("Std", "std"),
    ("Tot size", "tot_size"),
    ("IAT", "iat"),
    ("Number", "num"),
    ("Variance", "var"),
]

FEATURE_COLUMNS = [column for column, _ in FEATURE_PAIRS]
SERIALIZED_NAMES = [name for _, name in FEATURE_PAIRS]

# These columns travel with each row for traceability but must never enter a
# classifier or the textual serialization.
PROVENANCE_COLUMNS = ["SampleID", "SourceFile", "SourceRow", "FeatureGroup"]
TARGET_COLUMNS = ["Label", "Category"]

FEATURE_SET_EXCLUSIONS = {
    "original": frozenset(),
    "no_number": frozenset({"Number"}),
    "no_protocol": frozenset({"Protocol Type"}),
    "no_number_protocol": frozenset({"Number", "Protocol Type"}),
}


def feature_pairs(feature_set: str = "original") -> list[tuple[str, str]]:
    """Return the ordered columns for a registered feature-ablation setting."""
    try:
        excluded = FEATURE_SET_EXCLUSIONS[feature_set]
    except KeyError as exc:
        choices = ", ".join(FEATURE_SET_EXCLUSIONS)
        raise ValueError(f"unknown feature set {feature_set!r}; choose {choices}") from exc
    return [pair for pair in FEATURE_PAIRS if pair[0] not in excluded]


def format_value(value) -> str:
    """Match the exact numeric formatting used in textual model inputs."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    number = float(value)
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return f"{number:.4g}"


def serialize_frame(frame, selected_pairs=None):
    """Serialize a frame without depending on the CLI-oriented phase-02 script."""
    pairs = FEATURE_PAIRS if selected_pairs is None else selected_pairs
    columns = [column for column, _ in pairs]
    names = [name for _, name in pairs]
    values = frame[columns].to_numpy()
    return [
        " ".join(f"{name}={format_value(value)}" for name, value in zip(names, row))
        for row in values
    ]


def present_provenance(columns) -> list[str]:
    """Return provenance columns available in a legacy or revised artifact."""
    available = set(columns)
    return [column for column in PROVENANCE_COLUMNS if column in available]
