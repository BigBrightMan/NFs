# Open Questions

## Physics and data

- What detector-face `x,y` bounds and units are authoritative for every campaign?
- Does MuonDIS rate closure agree between FLUKA and Model 4 `global_c` samples?
- Which fixed tail thresholds should be reported consistently across years?

## Modeling

- Does [[Model 3]] provide value beyond [[Model 4]], especially for event-level weight studies?
- How stable are the selected Model 4 configurations across independent training seeds?
- Would a physics-aware factorization of momentum and space-time improve tails without increasing guard rejection?

## Validation

- Have reporting-only final-test artifacts been completed with one consistent guard version for every year?
- Should the 100,000-event MuonDIS delivery receive a separate validation report against the full weighted reference, distinct from the held-out test report?

## Infrastructure

- When should the existing FS code and outputs be frozen and the planned `NFs`, `NFs_data`, and `NFs_output` layout become canonical?
- Which legacy scripts are superseded but still needed to reproduce Model 1/2 results?
