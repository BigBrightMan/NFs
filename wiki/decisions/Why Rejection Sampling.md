---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/notes/REJECTION_SAMPLING_RUNBOOK.md
  - FS/src/flashsim/generation_guard.py
  - FS/scripts/create_guard_reject_v2.py
---

# Why Rejection Sampling

## Observation

An NF can generate finite points outside required physical, scoring-plane, detector, or catastrophic-tail constraints.

## Decision

Apply guards during generation. Reject failed candidates and draw replacements until the requested accepted count is reached or the finite attempt budget is exhausted.

## Current checks

- finite values, `E>10`, `pz>0`, and positive weights where relevant;
- positive mass-shell radicand and mass-shell tolerance;
- train-fitted scoring-plane tolerance;
- configured detector bounds for `x,y`;
- catastrophic-only robust bounds in transformed stable space.

In guard v2, `z` is constrained by the scoring-plane relation rather than an independent detector min/max.

## Decision gates

- below 0.1%: normal safety cleanup;
- 0.1–1%: inspect tails;
- 1–5%: model or guard needs review;
- above 5%: fail and do not deliver.

## Trade-off

Rejection sampling makes the delivered sample satisfy the declared contract but can hide a poor generator if the rejection fraction and reason counts are not reported.

Related validation: [[Tail Validation]], [[Validation Strategy]], and [[Model 4]].
