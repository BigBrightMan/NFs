---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/src/flashsim/model4.py
---

# Maximum Likelihood

Ordinary maximum likelihood minimizes the mean negative log probability:

$$
\mathcal{L}_{\mathrm{NLL}}=-\frac{1}{N}\sum_i \log p_\theta(x_i).
$$

[[Model 4]] instead uses a positive FLUKA-weighted objective:

$$
\mathcal{L}_{w\mathrm{NLL}}=-\frac{\sum_i w_i\log p_\theta(x_i)}{\sum_i w_i}.
$$

This changes the density learned by the model without making `w` a generated feature. See [[Weighted Density]] and [[FLUKA Weight]].
