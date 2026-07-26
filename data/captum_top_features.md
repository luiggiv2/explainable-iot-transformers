# DistilBERT — Layer Integrated Gradients (top features per class)

Captum LayerIG on the embedding layer; 120 correctly-classified test samples/class; n_steps 50; attributions remapped from wordpiece tokens to the 39 flow features via char offsets, L1-normalized per sample. signed_mean>0 pushes toward the class, <0 away.

## Benign
- `proto_num`  |attr|=0.1391  (+0.1389)
- `https`  |attr|=0.0952  (+0.0944)
- `ttl`  |attr|=0.0846  (+0.0658)
- `ack`  |attr|=0.0661  (-0.0440)
- `psh`  |attr|=0.0563  (+0.0463)
- `rate`  |attr|=0.0560  (+0.0492)
- `ack_cnt`  |attr|=0.0536  (+0.0530)
- `max`  |attr|=0.0460  (+0.0290)

## BruteForce
- `ssh`  |attr|=0.0972  (+0.0962)
- `proto_num`  |attr|=0.0812  (+0.0797)
- `psh`  |attr|=0.0737  (+0.0669)
- `ttl`  |attr|=0.0555  (+0.0530)
- `header_len`  |attr|=0.0548  (+0.0541)
- `min`  |attr|=0.0524  (+0.0515)
- `https`  |attr|=0.0481  (+0.0435)
- `tot_size`  |attr|=0.0480  (+0.0418)

## DDoS
- `num`  |attr|=0.1635  (+0.1635)
- `var`  |attr|=0.0648  (+0.0632)
- `syn_cnt`  |attr|=0.0502  (+0.0495)
- `max`  |attr|=0.0463  (+0.0384)
- `proto_num`  |attr|=0.0437  (+0.0094)
- `iat`  |attr|=0.0431  (-0.0025)
- `ack_cnt`  |attr|=0.0378  (+0.0305)
- `min`  |attr|=0.0359  (+0.0355)

## DoS
- `num`  |attr|=0.3367  (+0.3367)
- `var`  |attr|=0.1168  (+0.1114)
- `max`  |attr|=0.0658  (+0.0633)
- `iat`  |attr|=0.0520  (+0.0230)
- `header_len`  |attr|=0.0383  (+0.0338)
- `min`  |attr|=0.0352  (+0.0344)
- `avg`  |attr|=0.0317  (+0.0128)
- `tot_size`  |attr|=0.0307  (-0.0127)

## Mirai
- `tot_size`  |attr|=0.1826  (+0.1826)
- `max`  |attr|=0.1695  (+0.1694)
- `proto_num`  |attr|=0.1126  (+0.0693)
- `avg`  |attr|=0.1088  (+0.1088)
- `min`  |attr|=0.0833  (+0.0622)
- `num`  |attr|=0.0550  (+0.0267)
- `tot_sum`  |attr|=0.0494  (+0.0494)
- `iat`  |attr|=0.0475  (+0.0451)

## Recon
- `proto_num`  |attr|=0.1269  (+0.1269)
- `syn`  |attr|=0.0889  (+0.0815)
- `ack_cnt`  |attr|=0.0838  (+0.0820)
- `ttl`  |attr|=0.0651  (+0.0625)
- `header_len`  |attr|=0.0510  (+0.0501)
- `ack`  |attr|=0.0412  (-0.0202)
- `rst`  |attr|=0.0366  (+0.0326)
- `syn_cnt`  |attr|=0.0346  (+0.0329)

## Spoofing
- `proto_num`  |attr|=0.1164  (+0.1164)
- `max`  |attr|=0.1144  (+0.1120)
- `min`  |attr|=0.0946  (+0.0832)
- `ttl`  |attr|=0.0751  (+0.0716)
- `https`  |attr|=0.0465  (+0.0460)
- `header_len`  |attr|=0.0415  (+0.0415)
- `ack`  |attr|=0.0396  (-0.0306)
- `tot_size`  |attr|=0.0392  (+0.0269)

## Web
- `tot_size`  |attr|=0.1000  (+0.0996)
- `max`  |attr|=0.0817  (+0.0673)
- `avg`  |attr|=0.0813  (+0.0792)
- `ttl`  |attr|=0.0644  (+0.0576)
- `proto_num`  |attr|=0.0610  (+0.0516)
- `rate`  |attr|=0.0549  (+0.0458)
- `psh`  |attr|=0.0512  (+0.0386)
- `header_len`  |attr|=0.0500  (+0.0500)
