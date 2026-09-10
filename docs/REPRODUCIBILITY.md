# Reproducibility and audit definitions

## Protocol and provenance

`tabfm_dual_task_optimized.py` is the single modeling implementation for the study's TabFM ensemble and three comparator models. `run_benchmarks.py` launches that implementation with the configured dataset and records execution-environment metadata. `SOURCE_PROVENANCE.json` identifies the modeling source, study configuration and input schema; `PACKAGE_SHA256.json` records source-file hashes.

The study uses five-fold cross-validation for each task, with three-fold inner cross-validation for tuning. `config/manuscript.json` records these settings, seed 42, sensor-only predictors and 32 TabFM members. All models within a task use the same cross-validation splits.

Classical models jointly tune feature count and hyperparameters. An Extra Trees proxy chooses TabFM feature count within outer training data; the explanation stage evaluates the fitted TabFM ensemble itself. Search spaces are defined in the modeling script. Saved tuning JSON records the selected feature order and settings. Feature-audit CSVs are sorted by score and do not define model-input order.

An experiment record consists of the full run manifest, fold assignments, tuning records, feature/imputation audits, predictions, package versions, weight revision/hashes and device/dtype. Benchmark fold states have a checksum index. SHAP imports states after index validation and confirms prediction agreement. Serialized states must originate from a trusted experiment.

For outputs without indexed states, the runner refits each outer fold and checks predictions against saved outputs. Software or hardware changes can affect agreement; a mismatch blocks explanation. Benchmark outputs remain separate from the SHAP results directory.

## SHAP scope and checks

The supported method is `direct_full_ensemble_permutation_v1`. Each held-out observation is explained by its outer-fold model. Classification explains `probability_hard`; regression explains `predicted_performance`. All 32 fitted TabFM members are used. Selected training predictors are median-imputed using the outer training data and supplied in fitted order.

Defaults: 12 sampled training background rows, 10 permutation budget units, seed 42 and prediction batch size 32. With F selected features, `max_evals = 10 × (2F + 1)`. This is the SHAP permutation evaluation budget per explained row; it is not a count of GPU forward calls or a guarantee that all feature coalitions have been enumerated. The finite permutation/background approximation remains, even when every run is explained directly by the complete model.

Validation checks include:

- Expected held-out row identities and their unique fold assignment.
- Matching workbook and input-file hashes, fitted-state hashes, package versions and implementation/configuration fingerprints.
- Every selected feature represented once per completed run; finite values and matching model/task/fold metadata.
- Model prediction agreement with saved OOF outputs, using relative tolerance 1e-5 plus absolute tolerance 1e-5 for classification or 1e-3 for regression.
- The same output tolerances for the additive reconstruction `base value + sum(SHAP)` against the explained/saved prediction.

`results/<task>/models/tabfm_ensemble/shap_coverage.json` reports `direct_rows`, `total_rows`, missing row identities, errors and completion. The denominator is the expected OOF observations for that task, not the number of CSV feature rows. In the study-sized table it is 419 per task, or 838 task/run explanations across both tasks. Global plots are written after complete validated coverage. Features not selected by an individual fold have zero contribution when forming the global contribution matrix; they are not presented as observed feature values for that fold.

Study-level completeness and runtime are measured from the resulting coverage and progress files. Software tests use internal fixtures and known models; their results do not establish study-wide SHAP coverage or live LLM performance. The verification scope is described in [VALIDATION.md](VALIDATION.md).

## LLM gate and fallback accounting

The versioned prompt and exact reduced payload builder are in `narrative_explainer.py`. The prompt requests at most 180 words; the implemented length gate separately enforces at most 2,400 characters. Other gates check nonempty output, prohibited claim patterns, numeric tokens found in evidence, and basic SHAP/prediction context. These gates do not establish semantic completeness, correct causal reasoning, or correct assignment of every number to a feature. Manual review is still needed before claiming explanation fidelity.

`narrative_store.py` logs each new request outcome to `results/llm_narration_audit.jsonl` and caches accepted/gate-rejected results by evidence, prompt version, requested model and endpoint. Audit logs contain generated or rejected text and should be treated as study outputs. API credentials are not written by the application. Identical cache hits do not generate a second attempt record. Separate processes must not write the same cache concurrently.

`summarize_llm_audit.py` produces:

| Field | Definition |
| --- | --- |
| `attempts` | All logged generation attempts, including provider failures/unavailability |
| `unique_task_run_records` | Distinct task/Excel-row pairs in the log |
| `statuses` | Accepted, gate-rejected, request-failed or unavailable counts |
| `acceptance_fraction` | Accepted attempts / all logged attempts |
| `fallback_attempts`, `fallback_fraction` | Nonaccepted attempts and their fraction of all attempts |
| `checks.<name>.evaluated/passed/failed/pass_fraction` | Counts/rate among attempts where that check was evaluated |
| `failed_checks` | Failure counts by check; one attempt can fail multiple checks |
| `resolved_or_requested_models` | Resolved model where available, otherwise requested model |

Empty logs yield zero counts and null overall fractions. Provider failures without gate results do not enter per-check denominators. A gate failure or provider failure retains deterministic text. `llm_case_summary.json` from precomputation has a different denominator: processed task/run cases in that invocation, including cache hits. Attempt-level and case-level rates are therefore distinct measures.

Specific provider/model identifiers support experimental traceability; `openrouter/free` can resolve differently over time. Temperature zero does not make a remotely served model immutable. Accepted/rejected texts, timestamps, requested/resolved model fields and the prompt version form the generation record.
