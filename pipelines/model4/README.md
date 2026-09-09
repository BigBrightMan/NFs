# Model 4 pipeline

This directory contains declarative orchestration only. Scientific implementation lives in `src/flashsim_nf`; maintained scientific choices live in `configs`.

Stage files declare contracts and selection authority. Plan files choose a subset of stages and trials without editing generated job configurations.

The held-out test cannot run until model selection has produced a frozen `selected_model.json`.

The implemented baseline entry point is `scripts/create_model4_baseline_configs.py`. It resolves the maintained `baseline_only.yaml` intent into environment-specific immutable configs for selected years and preprocessing A/B/C. Each config must pass `scripts/preflight_model4_training.py` and is submitted individually with `scripts/submit_model4_training.py`.

After training, `scripts/evaluate_model4_training.py` creates optimization and
overfitting diagnostics without loading test data. After a physical generated
ROOT exists, `scripts/evaluate_model4_generated.py` computes weighted global,
marginal, and train-defined tail metrics against validation or a frozen final
test reference. See `wiki/validation/Post-training Evaluation.md`.
