# Layer-IG completeness audit

Captum's convergence residual is reported both in raw logit units and relative to the absolute target-logit difference between the observed input and the all-[PAD] content baseline.

- absolute residual: median 0.037147, mean 0.081207, 95th percentile 0.311077, maximum 1.585442;
- relative residual: median 0.60%, mean 1.23%, 95th percentile 4.45%, maximum 23.65%;
- rows at or below 5% relative residual: 96.0%;
- rows at or below 10% relative residual: 99.0%.

A large relative value can occur when the reference-to-input logit difference is close to zero, so the per-row audit should be consulted before deciding whether additional integration steps are warranted.
