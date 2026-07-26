# XGBoost — SHAP (top features per class)

TreeExplainer; 120 correctly-classified test samples/class; SHAP toward the true class, L1-normalized per sample (same protocol as the DistilBERT Layer-IG analysis). signed_mean>0 pushes toward the class.

## Benign
- `num`  |shap|=0.4027  (+0.4027)
- `https`  |shap|=0.1799  (+0.1729)
- `max`  |shap|=0.0964  (+0.0764)
- `psh`  |shap|=0.0392  (+0.0113)
- `rate`  |shap|=0.0368  (+0.0079)
- `ttl`  |shap|=0.0350  (+0.0179)
- `iat`  |shap|=0.0291  (+0.0154)
- `std`  |shap|=0.0258  (+0.0139)

## BruteForce
- `ssh`  |shap|=0.3297  (+0.2966)
- `num`  |shap|=0.2482  (+0.2482)
- `rate`  |shap|=0.0571  (+0.0490)
- `iat`  |shap|=0.0385  (+0.0352)
- `max`  |shap|=0.0340  (+0.0263)
- `syn`  |shap|=0.0298  (+0.0187)
- `https`  |shap|=0.0273  (+0.0164)
- `dns`  |shap|=0.0269  (+0.0197)

## DDoS
- `icmp`  |shap|=0.1972  (+0.1279)
- `avg`  |shap|=0.1693  (+0.1631)
- `psh`  |shap|=0.1024  (+0.0511)
- `fin`  |shap|=0.0957  (+0.0453)
- `iat`  |shap|=0.0623  (+0.0316)
- `num`  |shap|=0.0617  (+0.0617)
- `rate`  |shap|=0.0546  (+0.0357)
- `syn_cnt`  |shap|=0.0502  (+0.0337)

## DoS
- `num`  |shap|=0.2381  (+0.2381)
- `header_len`  |shap|=0.1941  (+0.1932)
- `syn`  |shap|=0.0727  (+0.0652)
- `iat`  |shap|=0.0620  (+0.0418)
- `avg`  |shap|=0.0576  (+0.0555)
- `fin`  |shap|=0.0550  (+0.0533)
- `rate`  |shap|=0.0480  (+0.0378)
- `psh`  |shap|=0.0448  (+0.0448)

## Mirai
- `tot_sum`  |shap|=0.6172  (+0.6172)
- `max`  |shap|=0.1023  (+0.1017)
- `proto_num`  |shap|=0.0868  (+0.0868)
- `min`  |shap|=0.0847  (+0.0795)
- `std`  |shap|=0.0557  (+0.0557)
- `avg`  |shap|=0.0191  (+0.0191)
- `ttl`  |shap|=0.0072  (-0.0064)
- `header_len`  |shap|=0.0048  (+0.0048)

## Recon
- `num`  |shap|=0.3168  (+0.3168)
- `tot_sum`  |shap|=0.1737  (+0.1691)
- `syn`  |shap|=0.1157  (+0.0988)
- `rst`  |shap|=0.0494  (+0.0372)
- `std`  |shap|=0.0409  (+0.0118)
- `http`  |shap|=0.0360  (+0.0237)
- `https`  |shap|=0.0335  (+0.0242)
- `fin`  |shap|=0.0330  (+0.0214)

## Spoofing
- `num`  |shap|=0.3376  (+0.3376)
- `min`  |shap|=0.2808  (+0.2751)
- `max`  |shap|=0.1209  (+0.1153)
- `iat`  |shap|=0.0413  (+0.0285)
- `https`  |shap|=0.0386  (-0.0007)
- `header_len`  |shap|=0.0323  (+0.0294)
- `avg`  |shap|=0.0211  (+0.0107)
- `std`  |shap|=0.0160  (-0.0015)

## Web
- `num`  |shap|=0.3798  (+0.3798)
- `iat`  |shap|=0.1094  (+0.1035)
- `max`  |shap|=0.0715  (+0.0460)
- `rate`  |shap|=0.0620  (+0.0546)
- `https`  |shap|=0.0492  (+0.0429)
- `http`  |shap|=0.0490  (+0.0340)
- `ttl`  |shap|=0.0344  (+0.0192)
- `std`  |shap|=0.0307  (+0.0217)
