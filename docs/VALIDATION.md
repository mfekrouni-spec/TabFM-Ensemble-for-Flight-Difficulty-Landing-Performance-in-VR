# Software verification

Verification environment: Windows, Python 3.11.2, 2026-09-10. Package versions are recorded in [requirements-environment-snapshot.txt](requirements-environment-snapshot.txt). Fresh dependency installation and Linux/macOS execution have not been tested.

## Automated software tests

Command: `python -m unittest discover -s tests -v` from the repository root.

**16 tests passed.** The tests use temporary internal fixtures, known models and mocked LLM responses.

| Check | Result |
| --- | --- |
| Known full-model classification SHAP contributions | Passed |
| Regression SHAP and interrupted-run resumption identity | Passed |
| Changed SHAP settings and partial records rejected | Passed |
| Approximation artifacts rejected as direct full-ensemble evidence | Passed |
| Saved prediction mismatch blocks explanations | Passed |
| Incomplete coverage not reported as complete | Passed |
| Reduced ensemble rejected for TabFM explanations | Passed |
| GUI evidence refresh, waterfall PNG, CSV export and cached LLM retrieval | Passed |
| LLM gate rejection retains deterministic text | Passed |
| Manuscript CLI preserves five outer folds, three inner folds and 32 members | Passed |
| Configuration overrides from another working directory | Passed |
| Workbook identity and integer-level contract | Passed |
| Automatic dtype selection respects device capability | Passed |
| LLM attempt/check denominators and empty-log fractions | Passed |
| Actual numeric-token gate rejects an invented number with a mocked provider | Passed |
| Benchmark resume guard rejects changed arguments | Passed |

Additional checks cover Python syntax, JSON syntax, CFF YAML parsing and four-author metadata, PowerShell launcher syntax, and configuration consistency with both study run manifests. All compared configuration fields, excluding task/model lists, matched. Source-file hashes and ZIP member hashes are generated and verified by `package_release.py`.

## Scope

The tests do not load the large TabFM checkpoints, reproduce study metrics, compute study-wide SHAP, contact a live LLM provider or validate upstream feature extraction. Optional browser screenshot export is outside the automated test coverage. CFF checks cover YAML parsing and author metadata rather than full external schema validation.

The study benchmark uses **five outer folds per task and three inner folds**. Study-level verification requires the tabular dataset, the complete modeling run, prediction-agreement checks, SHAP coverage records and LLM generation audits.
