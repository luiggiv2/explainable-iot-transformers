# RQ4 predicted-class control (reviewer issue #3b)

Attributing each input toward its OWN predicted class (perturbed -> its prediction), 60 correctly-classified rows/class, eps [0.01, 0.05, 0.1].

- coupling r (pred-preserved vs top-5 Jaccard) attributing toward PREDICTED class: **+0.841**
- for reference, attributing toward TRUE class (original analysis): +0.854
- top-5 Jaccard (pred-class) when prediction preserved: 0.354; when flipped: 0.237

| Class | pred preserved | top-5 Jaccard (pred-class) |
|---|---|---|
| Benign | 30.6% | 0.246 |
| BruteForce | 1.7% | 0.194 |
| DDoS | 88.3% | 0.316 |
| DoS | 47.8% | 0.377 |
| Mirai | 98.3% | 0.448 |
| Recon | 5.0% | 0.224 |
| Spoofing | 33.3% | 0.245 |
| Web | 46.1% | 0.255 |

Reading: if the predicted-class coupling stays close to the true-class value, the r=+0.85 is largely empirical rather than definitional; a large drop would indicate the true-class attribution inflated the low-rate fragility.
