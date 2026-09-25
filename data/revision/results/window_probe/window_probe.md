# Window-regime recoverability without `Number`

## Distribution of `Number` (packets per window)

| Split | Class | n | min | p01 | median | p99 | max | mean | share >= threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | Benign | 3315 | 8 | 10 | 10 | 10 | 10 | 10.00 | 0.0000 |
| train | BruteForce | 1071 | 9 | 10 | 10 | 10 | 10 | 10.00 | 0.0000 |
| train | DDoS | 18435 | 10 | 100 | 100 | 100 | 100 | 99.95 | 0.9995 |
| train | DoS | 8996 | 21 | 100 | 100 | 100 | 100 | 99.96 | 0.9998 |
| train | Mirai | 5132 | 3 | 100 | 100 | 100 | 100 | 99.74 | 0.9977 |
| train | Recon | 2627 | 10 | 10 | 10 | 10 | 10 | 10.00 | 0.0000 |
| train | Spoofing | 2206 | 9 | 10 | 10 | 10 | 10 | 10.00 | 0.0000 |
| train | Web | 1071 | 8 | 10 | 10 | 10 | 10 | 10.00 | 0.0000 |
| train | _ALL | 42853 | 3 | 10 | 100 | 100 | 100 | 78.33 | 0.7593 |
| test | Benign | 662 | 10 | 10 | 10 | 10 | 10 | 10.00 | 0.0000 |
| test | BruteForce | 214 | 10 | 10 | 10 | 10 | 10 | 10.00 | 0.0000 |
| test | DDoS | 3687 | 16 | 100 | 100 | 100 | 100 | 99.96 | 0.9995 |
| test | DoS | 1800 | 75 | 100 | 100 | 100 | 100 | 99.99 | 1.0000 |
| test | Mirai | 1026 | 10 | 100 | 100 | 100 | 100 | 99.82 | 0.9981 |
| test | Recon | 527 | 10 | 10 | 10 | 10 | 10 | 10.00 | 0.0000 |
| test | Spoofing | 441 | 10 | 10 | 10 | 10 | 10 | 10.00 | 0.0000 |
| test | Web | 215 | 3 | 10 | 10 | 10 | 10 | 9.97 | 0.0000 |
| test | _ALL | 8572 | 3 | 10 | 100 | 100 | 100 | 78.34 | 0.7593 |

Threshold justification: `Number` is bimodal at 10 and 100; only 41 of 42,853 train rows and 3 of 8,572 test rows lie strictly between 15 and 85, so any cut inside that gap, including 50, yields the same regime labels up to those rows.

## Depth-3 decision tree, regime (Number >= threshold) from the 38 no_number fields

- test accuracy 0.9951, balanced accuracy 0.9930 (42 errors in 8,572 rows);
- confusion (rows = true regime low/high, cols = predicted): [[2040, 23], [19, 6490]];
- regime defined by `Number` agrees with the category-level regime (DDoS/DoS/Mirai = 100-packet) on 0.9995 of test rows.

```
|--- iat <= 0.001
|   |--- header_len <= 22.780
|   |   |--- tot_sum <= 5987.500
|   |   |   |--- class: False
|   |   |--- tot_sum >  5987.500
|   |   |   |--- class: True
|   |--- header_len >  22.780
|   |   |--- ack_cnt <= 11.500
|   |   |   |--- class: False
|   |   |--- ack_cnt >  11.500
|   |   |   |--- class: True
|--- iat >  0.001
|   |--- tot_sum <= 32558.500
|   |   |--- ack_cnt <= 10.500
|   |   |   |--- class: False
|   |   |--- ack_cnt >  10.500
|   |   |   |--- class: True
|   |--- tot_sum >  32558.500
|   |   |--- avg <= 2348.730
|   |   |   |--- class: True
|   |   |--- avg >  2348.730
|   |   |   |--- class: False
```

## Depth-limited regression trees for `Number`

| Depth | test R^2 | test MAE (packets) |
|---:|---:|---:|
| 3 | 0.9733 | 0.931 |
| 6 | 0.9893 | 0.362 |

## Closed form

`Tot sum / AVG` equals `Number` (relative tolerance 1e-6) on 1.0000 of the 8,572 test rows with AVG != 0 (R^2 = 1.000000).
