# Data contract and access

The modeling input is the study's tabular feature dataset, distributed separately from the source code. Dataset access, use conditions and feature-extraction information are available from the corresponding author, Mohamed Fekrouni (m.fekrouni@uae.ac.ma). The repository contains the column schema but no participant observations or example dataset.

The raw recordings originate from [A multimodal dataset for investigating working memory in a virtual reality piloting task, version 1.0.0](https://physionet.org/content/virtual-reality-piloting/1.0.0/), DOI [10.13026/azwa-ge48](https://doi.org/10.13026/azwa-ge48). Access is requested through PhysioNet under the dataset's terms. The modeling feature table and the raw recordings are separate inputs.

## Workbook

Filename by default: `data/sensors.xlsx`. Sheet name: `data`. One header row, then one observation per run. The combined workflow requires these first five columns in order:

| Column | Type | Use |
| --- | --- | --- |
| `Subject` | finite integer | Pilot identifier; used for audits, GUI selection and cluster bootstrap |
| `level` | integer 1, 2, 3 or 4 | Difficulty label; levels 1–2 easy, 3–4 hard |
| `run` | finite integer | Run identifier within pilot/difficulty |
| `flight_hours` | finite numeric | Experience context; excluded from default predictors |
| `performance` | finite numeric | Regression target, in the units of the supplied table |

Every later column is a numeric predictor. The study table has **445 predictors**, 450 total columns and **419 observations**. [feature_schema.csv](../data/feature_schema.csv) contains header names, positions, roles and expected types only. Other numeric feature tables define separate experiments.

Each `Subject`/`level`/`run` combination must be unique. Missing/non-finite targets and missing identifiers are rejected. Numeric predictors may have missing values; median imputation is fitted inside training folds. Each selected predictor requires usable training-fold values. Column order and row order are part of the input contract: `source_excel_row` is the one-based Excel row including the header, so the first observation is row 2. Saved manifests identify the entire workbook by SHA-256. Changes to the workbook, including resaving it, require a new experiment directory.

`python validate_workbook.py` checks the feature-table contract. `python run_benchmarks.py --validate-only` additionally checks outer splits. Inner tuning is exercised during fitting, not by validation-only mode.

## Optional raw signals

Raw recordings are used for GUI traces only. The `raw_root` configuration identifies the authorized `task-ils` directory, preserving the dataset's subject/run layout and stream naming. Filename and stream conventions are defined in `raw_signal_loader.py`. Saved predictions and SHAP evidence remain available without raw recordings. The loader displays streams and does not compute the 445 modeling predictors.

## Processing scope

This repository implements modeling and interpretation from the feature table. The raw-to-feature extraction pipeline and a complete feature-unit dictionary are outside its scope. Reconstruction from raw recordings requires those additional materials; the column schema alone does not specify extraction algorithms or units.
