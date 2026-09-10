# TabFM Ensemble for Flight Difficulty and Landing Performance in VR

Research software for flight-difficulty classification and landing-performance regression from multimodal sensor features. The modeling implementation is [tabfm_dual_task_optimized.py](tabfm_dual_task_optimized.py), which evaluates a 32-member TabFM ensemble against XGBoost, random forest and k-nearest neighbors (KNN). Supporting modules provide run-level SHAP explanations, optional language-model summaries and a graphical review application.

**Authors:** Mohamed Fekrouni, Saad Chakkor, Mostafa Baghouri and Jawhar Laamech. [Affiliations and contacts](AUTHORS.md) · [Citation](CITATION.cff).

## Study protocol

| Component | Configuration |
| --- | --- |
| Classification target | Easy: difficulty levels 1–2; hard: levels 3–4 |
| Regression target | The feature table's `performance` column |
| Cross-validation | Five-fold cross-validation for each task |
| Inner cross-validation | Three folds within each outer training fold |
| Predictors | Sensor features; subject, difficulty level, run, flight hours and performance excluded |
| Preprocessing | Training-fold median imputation and feature selection; additional standardization for KNN |
| TabFM ensemble | 32 members |
| Random seed | 42 |

The full settings are recorded in [config/manuscript.json](config/manuscript.json). [Reproducibility documentation](docs/REPRODUCIBILITY.md) describes feature selection, persisted outputs and validation criteria.

## Data

The modeling workflow starts from the study's tabular feature dataset. The workbook is obtained separately from the corresponding author and is expected at `data/sensors.xlsx`, on a worksheet named `data`. [Data requirements](docs/DATA.md) describe access, column order and identifiers; [feature_schema.csv](data/feature_schema.csv) lists the 450 columns, including 445 predictors.

Raw recordings are available through the [PhysioNet dataset access process](https://physionet.org/content/virtual-reality-piloting/1.0.0/). They are optional inputs for GUI sensor traces. The repository does not contain participant observations, pretrained weights or the raw-to-feature extraction pipeline.

## Installation

The software tests were run on Windows with Python 3.11.2. Linux and macOS execution has not been verified. Full TabFM inference and all-run SHAP require substantial memory and computation; checkpoints are several GB per task.

From the repository root:

```powershell
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

The Linux/macOS activation command is `source .venv/bin/activate`. PyTorch is installed separately using its [official installation selector](https://pytorch.org/get-started/locally/) for the intended CPU/CUDA environment. The remaining dependencies are installed with:

```text
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) specifies dependency ranges. [The verification environment](docs/requirements-environment-snapshot.txt) records observed versions and is not a cross-platform installation lock or a complete record of the original experiment environment. An execution environment can be recorded with `python -m pip freeze > environment-used.txt`.

## Configuration and checkpoints

Defaults are defined in [workflow_config.json](workflow_config.json). An optional `workflow_config.local.json` overrides individual keys and is excluded from Git. `TABFM_CONFIG` can instead identify another JSON override file.

Relative data/output paths are resolved from the repository root. An explicit `TABFM_CONFIG` filename is resolved from the working directory. For example:

```json
{
  "workbook": "data/sensors.xlsx",
  "raw_root": "data/raw/task-ils",
  "device": "auto",
  "dtype": "auto",
  "checkpoint_revision": "main"
}
```

The official [Google TabFM checkpoints](https://huggingface.co/google/tabfm-1.0.0-pytorch) are downloaded with:

```text
python download_checkpoints.py
```

The downloader records the resolved Hub commit and task-file hashes in `classification_download.json` and `regression_download.json`. A fixed `checkpoint_revision` identifies an immutable weight version; the default `main` can change. Model weights are governed by their upstream license.

## Modeling

`run_benchmarks.py` launches the single modeling implementation, `tabfm_dual_task_optimized.py`, using the configured workbook and study settings.

```text
# Validate the feature table and outer splits without fitting
python run_benchmarks.py --validate-only

# Run all four models for classification and regression
python run_benchmarks.py
```

`--task classification` and `--task regression` select individual tasks. `--models` selects a model subset, for example `--models knn random_forest xgboost`. The SHAP/GUI workflow requires TabFM outputs for the corresponding task.

Outputs are written to `benchmark_results/tabfm_classification_optimized_results/` and the corresponding regression directory. They include manifests, fold assignments, split audits, tuning and feature-selection records, out-of-fold predictions, metrics, figures and TabFM fold states. The resume guard checks data, code, arguments and package versions. Experiments with changed settings or model lists require separate `source_root` and `results_root` directories.

## Run-level SHAP

```text
python run_all_shap.py --prepare-only
python run_all_shap.py
```

The explanation workflow evaluates each held-out run using its fitted 32-member outer-fold model. It checks saved-state provenance and prediction agreement before computation, saves completed observations, and resumes compatible results. A single-fold command is available:

```text
python run_all_shap.py --task classification --fold 1
```

Execution without `--fold` completes both tasks and generates global plots after validated coverage. Each output directory supports one writer at a time. Changed code, input data, weights, package versions, device/dtype or settings can invalidate persisted results.

The method is **direct full-ensemble permutation SHAP**. It uses every ensemble member and a finite permutation budget with a sampled training background. It does not use a surrogate or reduced-member model, and it does not enumerate all feature coalitions. Defaults are 12 background rows, seed 42 and `10 × (2F + 1)` evaluations per observation for F selected features. Classification explanations target the hard-class probability; regression explanations target the predicted performance value.

Coverage, missing rows, prediction agreement and additive reconstruction are reported in `results/<task>/models/tabfm_ensemble/shap_coverage.json` and the associated per-run files. [Audit definitions](docs/REPRODUCIBILITY.md) specify tolerances and denominators.

## Graphical application

After both task outputs have been prepared:

```text
python pilot_monitor_app.py
```

The GUI is served at `http://127.0.0.1:8766`. Task, pilot and run selection displays saved predictions, feature contributions, a waterfall plot and a narrative evidence card. Optional raw recordings add sensor traces. Missing SHAP is identified explicitly. CSV and waterfall PNG exports are supported; optional whole-page PNG export requires an installed Chromium browser.

Windows launchers are provided for the same workflow: `0_RUN_BENCHMARKS.cmd`, `1_COMPUTE_ALL_SHAP.cmd`, `2_OPEN_GUI.cmd` and `3_OPTIONAL_PRECOMPUTE_LLM.cmd`. They resolve Python from `TABFM_PYTHON`, the repository `.venv`, or the active `python` executable.

## Optional language-model summaries

Deterministic evidence cards are available without an API key. Optional LLM generation uses `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` from the process environment. `.env.example` lists the variables; `.env` files are not automatically loaded. The Windows launchers support a key prompt without saving the credential.

The default model alias, `openrouter/free`, can resolve to different models. A specific model identifier and the logged requested/resolved model fields support reproducible reporting. Requests transmit the reduced prediction and feature-contribution payload defined by `_detailed_llm_payload` in `narrative_explainer.py`; external processing remains subject to the dataset's use conditions.

After full SHAP coverage:

```text
# Generate a limited batch
python precompute_narratives.py --task classification --limit 2

# Generate summaries for all task/run combinations
python precompute_narratives.py

# Aggregate generation and fallback checks
python summarize_llm_audit.py
```

For 419 study observations, the unrestricted command covers 838 task/run combinations before cache reuse. Provider limits and charges depend on the selected service. Accepted summaries and gate-rejected fallbacks are cached by evidence, prompt version, requested model and endpoint. Provider failures stop batch processing and retain completed outputs.

The gate checks nonempty text, a character-length bound, prohibited claims, numeric tokens and core context. These are heuristic checks, not a complete semantic validation. Rejected or unavailable generation retains deterministic evidence text. `llm_audit_summary.json` reports attempt counts, evaluated/pass/fail counts for each check, acceptance/fallback fractions and model counts. Cache hits are excluded from new-attempt denominators.

## Verification

```text
python -m unittest discover -s tests -v
```

The [verification report](docs/VALIDATION.md) records 16 passing software tests. Tests use temporary internal fixtures, known models and mocked LLM responses. They do not run full TabFM checkpoints, reproduce study metrics or evaluate live LLM performance. Study-wide SHAP coverage and LLM acceptance rates are obtained from completed experiment outputs.

## Repository structure

| Files | Purpose |
| --- | --- |
| `tabfm_dual_task_optimized.py` | Modeling implementation for TabFM and all three comparator models |
| `run_benchmarks.py` | Modeling launcher and execution-environment checks |
| `direct_shap.py`, `run_all_shap.py` | Full-ensemble explanations, persistence and validation |
| `narrative_explainer.py`, `narrative_store.py` | Evidence text, optional LLM generation and caching |
| `precompute_narratives.py`, `summarize_llm_audit.py` | Batch generation and audit aggregation |
| `pilot_monitor_app.py`, `pilot_monitor_template.html`, `raw_signal_loader.py` | GUI and optional raw-signal display |
| `workflow_paths.py`, `workflow_config.json`, `config/`, launchers | Runtime paths and experiment settings |
| `download_checkpoints.py`, `validate_workbook.py` | Checkpoint setup and feature-table validation |
| `tests/`, `docs/`, `data/feature_schema.csv` | Software verification and data/method documentation |
| `SOURCE_PROVENANCE.json`, `PACKAGE_SHA256.json`, `package_release.py` | Source metadata and checksum-verified ZIP packaging |

## License and citation

Project code is distributed under the [MIT License](LICENSE); [NOTICE](NOTICE) identifies upstream components. TabFM source, model weights, research data and external services retain their respective terms. Citation metadata, author affiliations and ORCID identifiers are available in [CITATION.cff](CITATION.cff) and [AUTHORS.md](AUTHORS.md).
