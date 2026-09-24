# Layer-IG noise floor (eps = 0 control)

Each of the 120 inputs is drawn from the robustness cohort and attributed twice without any perturbation, using the same integration settings, field aggregation, normalization and metrics as the perturbation study, on the same device.

## Floor

| Metric | Value |
|---|---:|
| Mean Spearman over all 39 fields | 1.0000 |
| Mean Spearman over the top-8 fields | 1.0000 |
| Mean top-5 overlap | 5.00/5 |
| Bitwise-identical attribution vectors | 100.0% |
| Largest absolute difference in any field | 0.000e+00 |

## Floor compared with the perturbed measurements

| Condition | Mean Spearman | Mean top-5 overlap |
|---|---:|---:|
| eps = 0 (this control) | 1.000 | 5.00/5 |
| eps = 1% | 0.521 | 2.27/5 |
| eps = 5% | 0.476 | 2.03/5 |
| eps = 10% | 0.454 | 1.96/5 |

Repeated attribution of an unperturbed input reproduces the field vector exactly, so the floor is degenerate: every departure from it in the perturbation study is attributable to the changed input, not to the attribution method. This does not make individual rankings reliable — it only removes method noise as an explanation for their movement.
