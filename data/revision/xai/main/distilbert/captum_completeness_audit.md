# Layer-IG completeness audit

Captum's convergence residual is reported both in raw logit units and relative to the absolute target-logit difference between the observed input and the all-[PAD] content baseline.

- absolute residual: median 0.024855, mean 0.048675, 95th percentile 0.183800, maximum 1.185732;
- relative residual: median 0.39%, mean 0.72%, 95th percentile 2.54%, maximum 16.76%;
- rows at or below 5% relative residual: 99.2%;
- rows at or below 10% relative residual: 99.9%.

A large relative value can occur when the reference-to-input logit difference is close to zero, so the per-row audit should be consulted before deciding whether additional integration steps are warranted.
