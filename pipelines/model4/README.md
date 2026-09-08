# Model 4 pipeline

This directory contains declarative orchestration only. Scientific implementation lives in `src/flashsim_nf`; maintained scientific choices live in `configs`.

Stage files declare contracts and selection authority. Plan files choose a subset of stages and trials without editing generated job configurations.

The held-out test cannot run until model selection has produced a frozen `selected_model.json`.
