# Reproducibility audit

- Git commit: `34811ec5035c943d179e904e67f4e8e675c43a24`
- Data root: `data/revision`
- Rows: {'train': 42853, 'val': 8572, 'test': 8572}
- Provenance columns present: ['SampleID', 'SourceFile', 'SourceRow', 'FeatureGroup']

## Cross-split identity

| Comparison | Exact features | Exact features + subtype | Serialized `original` | Serialized `no_number` | Serialized `no_protocol` | Serialized `no_number_protocol` |
|---|---:|---:|---:|---:|---:|---:|
| val present in train | 0 | 0 | 0 | 0 | 0 | 0 |
| test present in train | 0 | 0 | 0 | 0 | 0 | 0 |
| test present in val | 0 | 0 | 0 | 0 | 0 | 0 |

## Duplicate and label-conflict groups

- Duplicate groups: 1690 (3658 rows).
- Groups crossing splits: 0.
- Conflicting subtype groups: 609.
- Conflicting category groups: 496.
- Largest exact-feature group: 9 rows.

## DistilBERT prediction consistency

- Accuracy reconstructed from predictions: 0.780098.
- Macro-F1 reconstructed from predictions: 0.710495.
- Confusion matrix identical: True.
- Absolute confusion-count difference: 0.
- Confusion cells changed: 0.
- SampleID order identical to test split: True.
- Parquet and NPY predictions identical: True.
- Parquet and split labels identical: True.

## Interpretation

Any non-zero cross-split feature/text identity is a leakage risk. A saved prediction array that does not reconstruct the reported confusion matrix must not be mixed with those reported metrics or downstream statistics.

Full SHA-256 checksums are stored in `revision_manifest.json`.
