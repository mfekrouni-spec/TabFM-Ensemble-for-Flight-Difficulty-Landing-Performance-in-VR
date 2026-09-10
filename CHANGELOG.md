# Changelog

## 1.0.0

- Modeling implementation in `tabfm_dual_task_optimized.py` for a 32-member TabFM ensemble, XGBoost, random forest and KNN.
- Five-fold cross-validation per task, three-fold inner tuning and seed 42.
- Portable configuration, checkpoint setup and feature-table validation.
- Direct full-ensemble permutation SHAP with per-run persistence, prediction/additivity checks, completeness accounting and global plots.
- Graphical review application with optional raw-signal traces, cached LLM summaries, deterministic fallbacks and generation audits.
- Software tests, experiment documentation, source checksums and author/citation metadata.
