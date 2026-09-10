"""Manually compute direct 32-member TabFM SHAP for every saved held-out run.

Uses the original run-level folds, selected-feature order and training medians.
All writes are confined to the configured results directory.
"""
import argparse
import gc
import importlib.metadata
import json
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from direct_shap import atomic_json, digest_file, output_column, predictor, write_summary, meta_path
from workflow_paths import ROOT, RESULTS, WORKBOOK, CONFIG, configured_path, task_dir, resolve_runtime

TASKS = ("classification", "regression")

def snapshot(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest_file(source) != digest_file(destination):
            raise ValueError(f"Source changed since preparation: {source}. Use a new results folder.")
    else:
        shutil.copy2(source, destination)

def prepare(task):
    source = configured_path("source_root") / f"tabfm_{task}_optimized_results"
    target = task_dir(task)
    if not (source / "run_manifest.json").is_file():
        raise FileNotFoundError(f"Benchmark artifacts missing at {source}. Run python run_benchmarks.py --task {task} first; see README.md.")
    if target == (source / task).resolve() or configured_path("source_root") in RESULTS.parents:
        raise ValueError("Choose a separate results folder; original study outputs must remain untouched")
    snapshot(source / "run_manifest.json", target / "source_run_manifest.json")
    for name in ("fold_assignments.csv", "all_models_oof_predictions.csv"):
        snapshot(source / task / name, target / name)
    model = Path("models/tabfm_ensemble")
    snapshot(source / task / model / "oof_predictions.csv", target / model / "oof_predictions.csv")
    for path in (source / task / model / "folds").glob("fold_*.csv"):
        if path.name.endswith(("_predictions.csv", "_selected_features.csv")):
            snapshot(path, target / model / "folds" / path.name)
    for path in (source / task / model / "folds").glob("fold_*_tuning.json"):
        snapshot(path, target / model / "folds" / path.name)
    # Fitted states are imported separately only after their index is verified.
    return target

def validate_inputs(task):
    target = task_dir(task)
    manifest = json.loads((target / "source_run_manifest.json").read_text(encoding="utf-8"))
    if manifest["split_protocol"] != "paper_sample" or int(manifest["tabfm_estimators"]) != 32:
        raise ValueError("Expected manuscript run-level protocol and 32 ensemble members")
    if digest_file(WORKBOOK) != manifest["input_sha256"]:
        raise ValueError("Workbook hash differs from the manuscript run manifest")
    frame = pd.read_excel(WORKBOOK, sheet_name=manifest["sheet"])
    folds = pd.read_csv(target / "fold_assignments.csv").sort_values("source_excel_row")
    wanted = set(range(2, len(frame) + 2))
    if len(frame) != int(manifest["rows"]) or set(folds.source_excel_row) != wanted or folds.source_excel_row.duplicated().any():
        raise ValueError("Invalid source-row/fold coverage")
    for column, name in (("subject", "subject_column"), ("level", "level_column"), ("performance", "performance_column")):
        if not np.allclose(folds[column].to_numpy(float), frame[manifest[name]].to_numpy(float), equal_nan=True):
            raise ValueError(f"Workbook and fold assignments disagree for {column}")
    if not np.array_equal(folds.difficulty_binary.astype(str), np.where(folds.level <= 2, "easy", "hard")):
        raise ValueError("Difficulty labels differ from manuscript mapping")
    fold_numbers = set(folds.test_fold.astype(int))
    if fold_numbers != set(range(1, int(manifest["folds"]) + 1)):
        raise ValueError("Unexpected outer fold identifiers")
    oof = pd.read_csv(target / "models/tabfm_ensemble/oof_predictions.csv").set_index("source_excel_row")
    all_oof = pd.read_csv(target / "all_models_oof_predictions.csv")
    all_oof = all_oof[all_oof.model.eq("tabfm_ensemble")].set_index("source_excel_row")
    if oof.index.duplicated().any() or set(oof.index) != wanted or all_oof.index.duplicated().any() or set(all_oof.index) != wanted:
        raise ValueError("Invalid TabFM OOF coverage")
    col = output_column(task)
    if not np.allclose(oof.sort_index()[col], all_oof.sort_index()[col], atol=1e-12, rtol=1e-12):
        raise ValueError("Model and GUI OOF predictions disagree")
    for fold in sorted(fold_numbers):
        train, test, indices, features = fold_inputs(target, frame, folds, manifest, fold)
        saved = pd.read_csv(target / "models/tabfm_ensemble/folds" / f"fold_{fold:02d}_predictions.csv").set_index("source_excel_row")
        expected_rows = set(folds.loc[folds.test_fold.eq(fold), "source_excel_row"])
        if saved.index.duplicated().any() or set(saved.index) != expected_rows or set(saved.fold) != {fold}:
            raise ValueError(f"Fold {fold} prediction rows differ from held-out assignments")
        if not np.allclose(saved.sort_index()[col], oof.loc[sorted(expected_rows), col], atol=1e-12, rtol=1e-12):
            raise ValueError("Fold predictions differ from combined OOF file")
        print(f"{task} fold {fold}: {len(train)} train / {len(test)} held out / {len(features)} features", flush=True)
    return frame, folds, manifest

def fold_inputs(target, frame, folds, manifest, fold):
    directory = target / "models/tabfm_ensemble/folds"
    tuning = json.loads((directory / f"fold_{fold:02d}_tuning.json").read_text(encoding="utf-8"))
    # The feature audit is sorted by score; the tuning record preserves model input order.
    features = tuning["selected_features"]
    if len(features) != len(set(features)) or len(features) != tuning["selected_feature_count"]:
        raise ValueError("Invalid selected-feature order")
    audit = pd.read_csv(directory / f"fold_{fold:02d}_selected_features.csv")
    selected = audit[audit.selected.astype(str).str.lower().isin(("true", "1"))].set_index("feature")
    if set(selected.index) != set(features) or selected.index.duplicated().any():
        raise ValueError("Tuning and selected-feature audit disagree")
    test_indices = folds.loc[folds.test_fold.eq(fold), "source_excel_row"].to_numpy(int) - 2
    train_indices = folds.loc[~folds.test_fold.eq(fold), "source_excel_row"].to_numpy(int) - 2
    values = frame[features].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    medians = values.iloc[train_indices].median()
    stored_medians = selected.loc[features, "training_imputation_median"].to_numpy(float)
    if not np.allclose(medians, stored_medians, atol=1e-10, rtol=1e-10):
        raise ValueError(f"Fold {fold}: training imputation medians differ from the source audit")
    train = values.iloc[train_indices].fillna(medians).reset_index(drop=True)
    test = values.iloc[test_indices].fillna(medians).reset_index(drop=True)
    if not np.isfinite(train.to_numpy()).all() or not np.isfinite(test.to_numpy()).all():
        raise ValueError("Non-finite imputed features")
    return train, test, (train_indices, test_indices), features

def compute_task(task, selected_folds=None):
    import joblib
    import shap
    import tabfm
    import torch
    import tabfm_dual_task_optimized as benchmark
    from direct_shap import explain_fold_with_shap, collect_validated

    target = task_dir(task)
    frame, folds, manifest = validate_inputs(task)
    seed = int(manifest["seed"])
    if int(CONFIG["seed"]) != seed:
        raise ValueError("Configured seed must match the manuscript seed")
    device, dtype = resolve_runtime(torch)
    environment = {name: importlib.metadata.version(name) for name in
                   ("numpy", "pandas", "shap", "torch", "tabfm", "scikit-learn", "joblib")}
    atomic_json(target / "shap_environment.json", dict(python=sys.version, packages=environment, device=device, dtype=str(dtype)))
    checkpoint = configured_path("checkpoint_root")
    checkpoint_files = [checkpoint / task / "config.json", checkpoint / task / "model.safetensors"]
    for path in checkpoint_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}. See README for the official model link.")
    print(f"Hashing {task} checkpoint for reproducibility...", flush=True)
    checkpoint_hashes = {p.name: digest_file(p) for p in checkpoint_files}
    backbone = None
    try:
        for fold in sorted(set(folds.test_fold.astype(int))):
            if selected_folds and fold not in selected_folds:
                continue
            directory = target / "models/tabfm_ensemble/folds"
            path = directory / f"fold_{fold:02d}_shap_values.csv"
            state_path = directory / f"fold_{fold:02d}_tabfm_fold_state.joblib"
            provenance_path = state_path.with_suffix(".meta.json")
            inputs = {p.name: digest_file(p) for p in [
                target / "fold_assignments.csv", target / "source_run_manifest.json",
                directory / f"fold_{fold:02d}_tuning.json",
                directory / f"fold_{fold:02d}_selected_features.csv",
                directory / f"fold_{fold:02d}_predictions.csv"]}
            context = dict(input_sha256=manifest["input_sha256"], source_files=inputs,
                checkpoint_hashes=checkpoint_hashes, packages=environment, seed=seed,
                device=device, dtype=str(dtype), members=32, tabfm_batch_size=int(manifest["tabfm_batch_size"]),
                runner_sha256=digest_file(__file__), benchmark_sha256=digest_file(benchmark.__file__))
            train, test, (train_indices, test_indices), features = fold_inputs(target, frame, folds, manifest, fold)
            state_meta = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else None
            # A fresh benchmark may provide a checksum-indexed fitted fold state.
            # States without an index are refitted and prediction-checked.
            source = configured_path("source_root") / f"tabfm_{task}_optimized_results"
            source_state = source / task / "models/tabfm_ensemble/folds" / state_path.name
            source_index = source / "state_index.json"
            if not state_path.exists() and source_state.exists() and source_index.exists():
                index = json.loads(source_index.read_text(encoding="utf-8"))
                if (index["run_manifest_sha256"] != digest_file(target / "source_run_manifest.json")
                    or index["benchmark_source_sha256"] != digest_file(benchmark.__file__)
                    or index["packages"] != environment or index["checkpoints"] != checkpoint_hashes
                    or index["states"].get(state_path.name) != digest_file(source_state)):
                    raise ValueError("Source fitted state does not match its checksum/environment index")
                shutil.copy2(source_state, state_path)
                state_meta = dict(context=context, state_sha256=digest_file(state_path))
                atomic_json(provenance_path, state_meta)
            if state_path.exists():
                if state_meta is None or state_meta["context"] != context or state_meta["state_sha256"] != digest_file(state_path):
                    raise ValueError(f"Incompatible fitted state for fold {fold}; choose a new results folder")
            elif path.exists():
                raise ValueError("SHAP exists without its fitted-state checkpoint")
            # Check all settings before skipping a completed fold, without loading the GPU.
            if path.exists() and meta_path(path).exists() and state_meta:
                metadata = json.loads(meta_path(path).read_text(encoding="utf-8"))
                expected_evals = (2 * len(features) + 1) * int(CONFIG["permutations"])
                if (metadata["fitted_context"] != state_meta or metadata["max_evals"] != expected_evals
                    or len(metadata["background_positions"]) != min(int(CONFIG["background_rows"]), len(train))
                    or metadata["prediction_batch_size"] != int(CONFIG["prediction_batch_size"])
                    or metadata["atol"] != CONFIG[f"{task}_atol"] or metadata["rtol"] != CONFIG["prediction_rtol"]
                    or metadata["implementation_sha256"] != digest_file(ROOT / "direct_shap.py")):
                    raise ValueError("SHAP settings changed; use a new results folder")
                values, summary = collect_validated(target)
                if summary["errors"]:
                    raise ValueError("; ".join(summary["errors"]))
                completed = set(values.loc[values.fold.eq(fold), "source_excel_row"].astype(int)) if not values.empty else set()
                if completed == set(test_indices + 2):
                    print(f"{task} fold {fold}: all direct SHAP records validated; skipped", flush=True)
                    continue
            if backbone is None:
                backbone = benchmark.load_tabfm_low_memory(task, str(checkpoint), device, dtype, torch)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if state_path.exists():
                estimator = benchmark.load_tabfm_fold_state(state_path, backbone, joblib)
            else:
                cls = tabfm.TabFMClassifier if task == "classification" else tabfm.TabFMRegressor
                estimator = cls.ensemble(model=backbone, n_estimators=32, max_num_features=500,
                    max_num_rows=None, batch_size=int(manifest["tabfm_batch_size"]), random_state=seed, verbose=False)
                target_values = np.where(frame[manifest["level_column"]] <= 2, "easy", "hard") if task == "classification" else frame[manifest["performance_column"]].to_numpy(float)
                with benchmark.progress_heartbeat(f"Fit {task} fold {fold} (32 members)", 1):
                    estimator.fit(train, target_values[train_indices])
                benchmark.save_tabfm_fold_state(estimator, state_path, joblib)
                state_meta = dict(context=context, state_sha256=digest_file(state_path))
                atomic_json(provenance_path, state_meta)
            predicted = predictor(estimator, task, features, int(CONFIG["prediction_batch_size"]))(test)
            expected = pd.read_csv(directory / f"fold_{fold:02d}_predictions.csv").set_index("source_excel_row").loc[test_indices + 2, output_column(task)].to_numpy(float)
            differences = np.abs(predicted - expected)
            matched = bool(np.allclose(predicted, expected, atol=CONFIG[f"{task}_atol"], rtol=CONFIG["prediction_rtol"]))
            atomic_json(directory / f"fold_{fold:02d}_prediction_check.json", dict(
                all_saved_predictions_match=matched, rows=len(test), max_absolute_difference=float(differences.max()),
                mean_absolute_difference=float(differences.mean()), atol=CONFIG[f"{task}_atol"], rtol=CONFIG["prediction_rtol"],
                fitted_state_sha256=state_meta["state_sha256"]))
            if not matched:
                raise RuntimeError(f"{task} fold {fold}: refitted outputs differ from manuscript predictions "
                    f"(max {differences.max():.6g}). Prediction-check JSON saved. Original predictions remain unchanged; "
                    "resolve model/environment provenance before pairing SHAP with these predictions.")
            args = SimpleNamespace(seed=seed, shap_method="permutation", shap_explain_rows=0,
                shap_source_rows=None, shap_background_size=int(CONFIG["background_rows"]),
                shap_permutations=int(CONFIG["permutations"]), shap_max_evals=0,
                shap_batch_size=int(CONFIG["prediction_batch_size"]), shap_checkpoint_rows=1,
                shap_context_fingerprint=state_meta, shap_atol=CONFIG[f"{task}_atol"], shap_rtol=CONFIG["prediction_rtol"])
            with benchmark.progress_heartbeat(f"Direct SHAP {task} fold {fold}", 1):
                explain_fold_with_shap(task, "tabfm_ensemble", fold, estimator, train, test, test_indices,
                    np.arange(2, len(frame) + 2), frame[manifest["subject_column"]].to_numpy(), path, args, shap)
            print(write_summary(target), flush=True)
            del estimator
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        del backbone
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    summary = write_summary(target, plots=True)
    print(f"{task}: {summary['direct_rows']}/{summary['total_rows']} valid direct SHAP records", flush=True)
    if not selected_folds and not summary["complete"]:
        raise RuntimeError("Incomplete SHAP coverage; inspect shap_coverage.json")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "both"), default="both")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fold", nargs="+", type=int, choices=range(1, 6))
    args = parser.parse_args()
    for name in ("background_rows", "permutations", "prediction_batch_size"):
        if not isinstance(CONFIG[name], int) or CONFIG[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for task in TASKS if args.task == "both" else (args.task,):
        prepare(task)
        if args.prepare_only or args.validate_only:
            validate_inputs(task)
            summary = write_summary(task_dir(task))
            print(f"{task}: inputs validated; {summary['direct_rows']}/{summary['total_rows']} SHAP rows ready")
        else:
            compute_task(task, args.fold)

if __name__ == "__main__":
    main()
