# Layer-IG completeness audit

Captum's convergence residual is reported both in raw logit units and relative to the absolute target-logit difference between the observed input and the all-[PAD] content baseline.

- absolute residual: median 0.022530, mean 0.040815, 95th percentile 0.136579, maximum 0.410674;
- relative residual: median 0.38%, mean 0.62%, 95th percentile 2.17%, maximum 6.76%;
- rows at or below 5% relative residual: 99.6%;
- rows at or below 10% relative residual: 100.0%.

A large relative value can occur when the reference-to-input logit difference is close to zero, so the per-row audit should be consulted before deciding whether additional integration steps are warranted.
