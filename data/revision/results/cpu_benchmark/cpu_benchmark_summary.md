# Controlled CPU inference benchmark

All models use the same 512 rows, 4 CPU threads, 1 warm-up pass(es), and 5 measured passes. Values are medians across measured passes; brackets show the interquartile range.

| Model | Batch | Timing boundary | ms/sample | samples/s | Size (MB) |
|---|---:|---|---:|---:|---:|
| xgb | 1 | model_only | 0.1734 [0.1722, 0.1735] | 5767.1 | 1.54 |
| svm | 1 | model_only | 0.9459 [0.9433, 0.9491] | 1057.2 | 0.99 |
| rf | 1 | model_only | 12.5434 [12.5313, 12.5569] | 79.7 | 61.51 |
| distilbert | 1 | model_only | 21.3263 [21.1690, 21.5781] | 46.9 | 268.56 |
| xgb | 1 | with_input_preparation | 0.3766 [0.3764, 0.3783] | 2655.4 | 1.54 |
| svm | 1 | with_input_preparation | 1.1296 [1.1294, 1.1297] | 885.3 | 0.99 |
| rf | 1 | with_input_preparation | 12.8125 [12.8027, 12.8282] | 78.0 | 61.51 |
| distilbert | 1 | with_input_preparation | 22.4015 [22.3052, 22.4501] | 44.6 | 268.56 |
| xgb | 64 | model_only | 0.0092 [0.0091, 0.0096] | 108607.8 | 1.54 |
| rf | 64 | model_only | 0.3992 [0.3977, 0.4021] | 2504.8 | 61.51 |
| svm | 64 | model_only | 0.8259 [0.8253, 0.8312] | 1210.9 | 0.99 |
| distilbert | 64 | model_only | 19.6898 [19.6864, 20.4016] | 50.8 | 268.56 |
| xgb | 64 | with_input_preparation | 0.0122 [0.0122, 0.0124] | 81825.1 | 1.54 |
| rf | 64 | with_input_preparation | 0.4092 [0.4072, 0.4102] | 2443.8 | 61.51 |
| svm | 64 | with_input_preparation | 0.8285 [0.8282, 0.8293] | 1207.0 | 0.99 |
| distilbert | 64 | with_input_preparation | 20.1542 [20.0986, 20.1835] | 49.6 | 268.56 |

`model_only` excludes tokenization or DataFrame-to-array conversion. `with_input_preparation` includes those operations but assumes that a serialized flow string or numeric feature row already exists. These measurements characterize this machine and software stack; they are not hardware-independent latency claims.

The encoder and the classical models were timed in separate, sequentially executed processes because PyTorch and XGBoost cannot share one process on this platform. Each family therefore held the declared thread budget alone, and no measurement overlapped another.
