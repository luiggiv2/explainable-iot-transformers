# Registered feature ablations and shortcut diagnostics

All results use the leakage-safe grouped split. Tree ablations change only the registered input columns; hyperparameters and test rows remain fixed.

## Macro-F1

| feature_set        |     rf |      svm |    xgb |
|:-------------------|-------:|---------:|-------:|
| no_number          | 0.7225 | nan      | 0.7239 |
| no_number_protocol | 0.7232 | nan      | 0.7208 |
| no_protocol        | 0.7236 | nan      | 0.7287 |
| original           | 0.7223 |   0.6672 | 0.7278 |

## Macro-F1 delta from the original feature set

| feature_set        |     rf |   svm |     xgb |
|:-------------------|-------:|------:|--------:|
| no_number          | 0.0002 |   nan | -0.0039 |
| no_number_protocol | 0.0009 |   nan | -0.007  |
| no_protocol        | 0.0013 |   nan |  0.001  |
| original           | 0      |     0 |  0      |

The pre-registered expansion gate (absolute macro-F1 change >= 0.01) was not triggered for the tree models.

## Simple shortcut diagnostics

| diagnostic             | features              |   accuracy |   macro_f1 |
|:-----------------------|:----------------------|-----------:|-----------:|
| class-prior dummy      | none                  |     0.4301 |     0.0752 |
| Number only            | Number                |     0.5073 |     0.1512 |
| Protocol Type only     | Protocol Type         |     0.5233 |     0.2288 |
| Number + Protocol Type | Number, Protocol Type |     0.6093 |     0.3561 |

These depth-3 trees measure how much label information is available from the suspect variables alone; they are diagnostics, not competitive baselines.

## `Number` distribution by category (test split)

| Category   |   count |   mean |   min |   max |   nunique |
|:-----------|--------:|-------:|------:|------:|----------:|
| Benign     |     662 | 10     |    10 |    10 |         1 |
| BruteForce |     214 | 10     |    10 |    10 |         1 |
| DDoS       |    3687 | 99.957 |    16 |   100 |         3 |
| DoS        |    1800 | 99.986 |    75 |   100 |         2 |
| Mirai      |    1026 | 99.825 |    10 |   100 |         2 |
| Recon      |     527 | 10     |    10 |    10 |         1 |
| Spoofing   |     441 | 10     |    10 |    10 |         1 |
| Web        |     215 |  9.967 |     3 |    10 |         2 |

## Short/long-window regime fraction by category

| Category   |   long (>=50) |   short (<50) |
|:-----------|--------------:|--------------:|
| Benign     |         0     |         1     |
| BruteForce |         0     |         1     |
| DDoS       |         0.999 |         0.001 |
| DoS        |         1     |         0     |
| Mirai      |         0.998 |         0.002 |
| Recon      |         0     |         1     |
| Spoofing   |         0     |         1     |
| Web        |         0     |         1     |

## Largest per-class ablation changes

| feature_set        | model   | class      |     f1 |   delta_vs_original |
|:-------------------|:--------|:-----------|-------:|--------------------:|
| no_number_protocol | xgb     | Web        | 0.4136 |             -0.0269 |
| no_number_protocol | rf      | Web        | 0.4463 |              0.0231 |
| no_protocol        | rf      | Web        | 0.44   |              0.0168 |
| no_number          | xgb     | Web        | 0.4241 |             -0.0164 |
| no_number_protocol | xgb     | BruteForce | 0.5393 |             -0.0128 |
| no_number_protocol | xgb     | Recon      | 0.6508 |             -0.011  |
| no_number_protocol | rf      | BruteForce | 0.5515 |             -0.0089 |
| no_number          | rf      | Recon      | 0.6919 |              0.0085 |
| no_number          | xgb     | Recon      | 0.6532 |             -0.0085 |
| no_protocol        | xgb     | BruteForce | 0.5585 |              0.0064 |
| no_protocol        | rf      | Recon      | 0.6892 |              0.0059 |
| no_number          | rf      | DoS        | 0.6376 |             -0.0058 |
