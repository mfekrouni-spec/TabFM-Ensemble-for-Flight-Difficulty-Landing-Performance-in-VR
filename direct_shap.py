"""Direct permutation SHAP of fitted models, with one atomic checkpoint per run.

The prediction function is the full fitted ensemble (including calibration).
Finite permutation sampling estimates Shapley values; no surrogate is fitted.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

METHOD = "direct_full_ensemble_permutation_v1"

def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)

def atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)

def meta_path(path):
    return Path(path).with_suffix(".meta.json")

def output_column(task):
    return "probability_hard" if task == "classification" else "predicted_performance"

def predictor(estimator, task, features, batch_size):
    model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    if task == "classification":
        classes = list(model.classes_)
        matches = [i for i, c in enumerate(classes) if str(c).casefold() == "hard" or c == 1]
        if len(matches) != 1:
            raise ValueError(f"Cannot identify hard-class output in {classes}")
        column = matches[0]

    def predict(values):
        frame = pd.DataFrame(np.asarray(values), columns=features)
        parts = []
        for start in range(0, len(frame), batch_size):
            block = frame.iloc[start:start + batch_size]
            if task == "classification":
                result = np.asarray(model.predict_proba(block), dtype=float)[:, column]
            else:
                result = np.asarray(model.predict(block), dtype=float).reshape(-1)
            if not np.isfinite(result).all():
                raise ValueError("Non-finite model output during direct SHAP")
            parts.append(result)
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    return predict

def validate_group(group, metadata, expected=None):
    """Reject partial, stale, duplicate or numerically invalid run records."""
    features = metadata["features"]
    required = {"feature", "source_excel_row", "fold", "model", "task", "feature_value",
                "shap_value", "base_value", "explanation_method", "context_signature",
                "ensemble_members", "max_evals", "background_rows", "shap_seed",
                output_column(metadata["task"])}
    if required.difference(group.columns):
        raise ValueError(f"Incomplete SHAP schema: {sorted(required.difference(group.columns))}")
    if len(group) != len(features) or group.feature.duplicated().any() or set(group.feature) != set(features):
        raise ValueError("SHAP record has missing, duplicate or unexpected features")
    for column, value in (("fold", metadata["fold"]), ("model", metadata["model"]),
                          ("task", metadata["task"]), ("explanation_method", METHOD),
                          ("context_signature", metadata["signature"]),
                          ("ensemble_members", metadata["ensemble_members"]),
                          ("max_evals", metadata["max_evals"]),
                          ("background_rows", len(metadata["background_positions"]))):
        if set(group[column]) != {value}:
            raise ValueError(f"SHAP metadata mismatch: {column}")
    if group.source_excel_row.nunique() != 1:
        raise ValueError("Expected exactly one run")
    row = int(group.source_excel_row.iloc[0])
    if row not in metadata["test_source_rows"]:
        raise ValueError("SHAP run is outside its held-out fold")
    if set(group.shap_seed) != {int((metadata["seed"] + 1000003 * metadata["fold"] + row) % (2**32))}:
        raise ValueError("SHAP seed mismatch")
    col = output_column(metadata["task"])
    if not np.isfinite(group[["feature_value", "shap_value", "base_value", col]].to_numpy(float)).all():
        raise ValueError("Non-finite SHAP record")
    if group.base_value.nunique() != 1 or group[col].nunique() != 1:
        raise ValueError("Inconsistent baseline/output within a run")
    prediction = float(group[col].iloc[0])
    residual = abs(float(group.base_value.iloc[0] + group.shap_value.sum()) - prediction)
    tolerance = metadata["atol"] + metadata["rtol"] * abs(prediction)
    if residual > tolerance:
        raise ValueError(f"SHAP additivity residual {residual:.6g} exceeds {tolerance:.6g}")
    if expected is not None:
        if int(expected["fold"]) != metadata["fold"]:
            raise ValueError("Prediction and SHAP fold differ")
        delta = abs(prediction - float(expected[col]))
        if delta > metadata["atol"] + metadata["rtol"] * abs(float(expected[col])):
            raise ValueError(f"SHAP output differs from saved OOF prediction by {delta:.6g}")
    return residual

def explain_fold_with_shap(task, model_key, fold_number, estimator, train_input,
                           test_input, test_indices, source_excel_rows, subjects,
                           output_path, args, shap, pd=pd, np=np):
    from tabfm_dual_task_optimized import assign_modality
    output_path = Path(output_path)
    features = list(map(str, test_input.columns))
    model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    members = int(getattr(model, "n_estimators", 1))
    if model_key == "tabfm_ensemble" and members != 32:
        raise ValueError(f"This release requires the full 32-member ensemble; found {members}")
    if getattr(args, "shap_method", "permutation") != "permutation":
        raise ValueError("Only direct permutation SHAP is supported")
    if not len(features) or not np.isfinite(train_input.to_numpy(float)).all() or not np.isfinite(test_input.to_numpy(float)).all():
        raise ValueError("SHAP inputs must contain finite, imputed selected features")
    background_count = min(int(args.shap_background_size), len(train_input))
    if background_count < 1:
        raise ValueError("Background sample must be nonempty")
    positions = np.sort(np.random.default_rng(args.seed + 1000 * fold_number).choice(
        len(train_input), size=background_count, replace=False))
    minimum = 2 * len(features) + 1
    max_evals = int(args.shap_max_evals or minimum * getattr(args, "shap_permutations", 10))
    if max_evals < minimum:
        raise ValueError(f"Need at least {minimum} evaluations")
    test_rows = [int(source_excel_rows[int(i)]) for i in test_indices]
    data_hash = hashlib.sha256()
    for data in (train_input, test_input):
        data_hash.update(np.ascontiguousarray(data.to_numpy(dtype="<f8")).tobytes())
    atol = float(getattr(args, "shap_atol", 1e-5 if task == "classification" else 1e-3))
    rtol = float(getattr(args, "shap_rtol", 1e-5))
    metadata = dict(schema=1, method=METHOD, task=task, model=model_key, fold=int(fold_number),
        features=features, ensemble_members=members, background_positions=positions.tolist(),
        test_source_rows=test_rows, seed=int(args.seed), max_evals=max_evals,
        minimum_forward_reverse_cycles=max_evals // minimum,
        prediction_batch_size=int(args.shap_batch_size), atol=atol, rtol=rtol,
        data_sha256=data_hash.hexdigest(),
        fitted_context=getattr(args, "shap_context_fingerprint", "benchmark_fitted_state"),
        implementation_sha256=digest_file(__file__), shap_version=shap.__version__,
        interpretation="Direct full-model permutation SHAP estimates; no surrogate or reduced ensemble")
    metadata["signature"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    mp = meta_path(output_path)
    if mp.exists() and json.loads(mp.read_text(encoding="utf-8")) != metadata:
        raise ValueError(f"SHAP configuration or fitted context changed: {mp}. Use a new results directory.")
    if output_path.exists() and not mp.exists():
        raise ValueError(f"Unverified legacy SHAP checkpoint: {output_path}. Use a new results directory.")
    atomic_json(mp, metadata)
    saved = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    prediction_path = output_path.with_name(output_path.name.replace("_shap_values.csv", "_predictions.csv"))
    expected = pd.read_csv(prediction_path).set_index("source_excel_row") if prediction_path.exists() else None
    done = set()
    if not saved.empty:
        for row, group in saved.groupby("source_excel_row"):
            validate_group(group, metadata, expected.loc[int(row)] if expected is not None else None)
            done.add(int(row))
    requested = list(range(len(test_input)))
    if getattr(args, "shap_source_rows", None):
        requested = [i for i in requested if test_rows[i] in args.shap_source_rows]
    elif args.shap_explain_rows:
        requested = sorted(np.random.default_rng(args.seed + fold_number).choice(
            len(test_input), min(args.shap_explain_rows, len(test_input)), replace=False).tolist())
    predict = predictor(estimator, task, features, int(args.shap_batch_size))
    masker = shap.maskers.Independent(train_input.iloc[positions], max_samples=background_count)
    started = time.monotonic()
    newly_done = 0
    print(f"Fold {fold_number}: {len(done)}/{len(requested)} runs saved; {members} members, {max_evals} evaluations per run", flush=True)
    for position in requested:
        row = test_rows[position]
        if row in done:
            continue
        row_seed = int((args.seed + 1000003 * fold_number + row) % (2**32))
        explainer = shap.Explainer(predict, masker, algorithm="permutation", feature_names=features, seed=row_seed)
        data = test_input.iloc[[position]]
        print(f"  Explaining Excel row {row} ({len(done) + 1}/{len(requested)})...", flush=True)
        explanation = explainer(data, max_evals=max_evals, batch_size=args.shap_batch_size, silent=True)
        values = np.asarray(explanation.values, dtype=float).reshape(1, len(features))[0]
        base = float(np.asarray(explanation.base_values).reshape(-1)[0])
        output = float(predict(data)[0])
        frame = pd.DataFrame(dict(task=task, model=model_key, fold=fold_number, source_excel_row=row,
            subject=subjects[int(test_indices[position])], feature=features,
            modality=[assign_modality(f) for f in features], feature_value=data.to_numpy(float)[0],
            shap_value=values, base_value=base, **{output_column(task): output},
            explanation_method=METHOD, ensemble_members=members, background_rows=background_count,
            max_evals=max_evals, shap_seed=row_seed, context_signature=metadata["signature"]))
        residual = validate_group(frame, metadata, expected.loc[row] if expected is not None else None)
        frame["reconstruction_residual"] = residual
        saved = pd.concat([saved, frame], ignore_index=True)
        atomic_csv(output_path, saved)
        done.add(row)
        newly_done += 1
        elapsed = time.monotonic() - started
        atomic_json(output_path.with_suffix(".progress.json"), dict(completed_rows=sorted(done),
            expected_rows=test_rows, complete=set(test_rows) == done, elapsed_seconds=elapsed))
        print(f"  Saved {len(done)}/{len(requested)}; residual={residual:.3g}; estimated remaining "
              f"{elapsed / newly_done * len(set(test_rows) - done) / 3600:.2f} h", flush=True)

def collect_validated(task_dir):
    """Read direct fold files only; aggregates never substitute for missing files."""
    task_dir = Path(task_dir)
    model_dir = task_dir / "models" / "tabfm_ensemble"
    predictions = pd.read_csv(model_dir / "oof_predictions.csv")
    if predictions.source_excel_row.duplicated().any():
        raise ValueError("Duplicate OOF rows")
    expected = predictions.set_index("source_excel_row")
    frames, errors = [], []
    seen = set()
    for path in sorted((model_dir / "folds").glob("fold_*_shap_values.csv")):
        try:
            metadata = json.loads(meta_path(path).read_text(encoding="utf-8"))
            if metadata["ensemble_members"] != 32 or metadata["method"] != METHOD:
                raise ValueError("Not direct 32-member ensemble SHAP")
            frame = pd.read_csv(path)
            for row, group in frame.groupby("source_excel_row"):
                row = int(row)
                if row in seen or row not in expected.index:
                    raise ValueError("Duplicate or unexpected SHAP row")
                validate_group(group, metadata, expected.loc[row])
                seen.add(row)
                frames.append(group)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    values = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    wanted = set(expected.index.astype(int))
    summary = dict(direct_rows=len(seen), total_rows=len(wanted), complete=seen == wanted and not errors,
                   missing_source_rows=sorted(wanted - seen), errors=errors,
                   method=METHOD, surrogate_used=False)
    return values, summary

def write_summary(task_dir, plots=False):
    task_dir = Path(task_dir)
    values, summary = collect_validated(task_dir)
    model_dir = task_dir / "models" / "tabfm_ensemble"
    atomic_json(model_dir / "shap_coverage.json", summary)
    if not values.empty:
        atomic_csv(model_dir / "shap_values_all_explained_rows.csv", values)
        audits = values.groupby("source_excel_row").agg(fold=("fold", "first"),
            features=("feature", "count"), reconstruction_residual=("reconstruction_residual", "first"))
        atomic_csv(model_dir / "shap_run_audit.csv", audits.reset_index())
    if summary["complete"]:
        atomic_csv(model_dir / "oof_shap_values.csv", values)
        if plots:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import shap
            matrix = values.pivot(index="source_excel_row", columns="feature", values="shap_value").fillna(0)
            data = values.pivot(index="source_excel_row", columns="feature", values="feature_value").reindex(columns=matrix.columns)
            # Features absent from a fold model contribute zero to that model.
            importance = matrix.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").reset_index()
            atomic_csv(model_dir / "shap_feature_importance.csv", importance)
            for kind in ("bar", "beeswarm"):
                explanation = shap.Explanation(matrix.to_numpy(), data=data.to_numpy(), feature_names=list(matrix.columns))
                getattr(shap.plots, kind)(explanation, max_display=20, show=False)
                plt.gcf().set_size_inches(12, 9)
                plt.title("Direct full-ensemble SHAP across all held-out runs")
                plt.tight_layout()
                plt.savefig(model_dir / f"figure_shap_{kind}.png", dpi=220, bbox_inches="tight")
                plt.close("all")
    return summary
