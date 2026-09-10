#!/usr/bin/env python
r"""
Optimized dual-task pilot-sensor benchmark using the original workbook.

Tasks
-----
1. Binary difficulty classification: levels 1-2 = easy, 3-4 = hard.
2. Continuous performance regression: predict the measured ``performance``.

Models
------
The official TabFM ensemble is benchmarked against tuned XGBoost, Random
Forest, and K-nearest-neighbour pipelines. Every model receives the same outer
folds. Within each outer training fold, a separate inner cross-validation
search selects a smaller predictor set and hyperparameters without seeing the
outer test rows. KNN additionally uses training-fold standardization.

Classical models jointly tune mutual-information feature count and model
hyperparameters. Because repeatedly tuning the large TabFM backbone would be
prohibitively expensive, an Extra Trees proxy selects TabFM's feature count
inside the outer training fold; the official TabFM ensemble then receives only
those selected predictors. Feature selections, inner-CV scores, and best
settings are saved for every outer fold.

The default feature set is ``sensors_only``: Subject, level, run,
flight_hours, and performance are treated as identifiers/targets/context and
excluded from the predictors. ``sensors_plus_context`` adds run and
flight_hours to both tasks and adds level to regression. Performance is never
used to predict difficulty, and level is never used to predict performance in
the default leakage-safe configuration.

The default ``paper_sample`` split follows the project's paper-style protocol:
sample-level five-fold CV with pilots allowed in both train and test.
``grouped_pilot`` is available for the stricter unseen-pilot question.

Fold predictions are saved immediately and compatible runs resume
automatically. Pilot-cluster bootstrap confidence intervals and paired
differences quantify uncertainty. The script never modifies the input Excel
file.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import threading
import time
from typing import Any


from workflow_paths import ROOT as DEFAULT_ROOT, WORKBOOK, RESULTS, configured_path
DEFAULT_INPUT = WORKBOOK
DEFAULT_CHECKPOINT = configured_path("checkpoint_root")
DEFAULT_OUTPUT = RESULTS / "benchmark_rerun"

MODEL_CHOICES = ("tabfm_ensemble", "xgboost", "random_forest", "knn")
TASK_CHOICES = ("classification", "regression")
MODEL_DISPLAY = {
    "tabfm_ensemble": "TabFM Ensemble",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "knn": "KNN",
}
TASK_DISPLAY = {
    "classification": "Difficulty classification",
    "regression": "Performance regression",
}
MODEL_COLORS = {
    "tabfm_ensemble": "#0072B2",
    "xgboost": "#D55E00",
    "random_forest": "#009E73",
    "knn": "#7A3E9D",
}
MODALITY_ORDER = [
    "ECG/HRV",
    "EDA",
    "PPG",
    "EMG",
    "Eye Movement",
    "Respiration",
    "Forearm Accel.",
    "Torso Accel.",
    "Head Movement",
    "Context",
    "Other",
]
HEAD_MOVEMENT_SET = {
    "ang_vel_mean", "ang_vel_std", "ang_vel_max", "ang_acc_mean",
    "ang_acc_std", "ang_jerk_mean", "motion_smoothness",
    "angular_speed_mean", "angular_speed_std", "fixation_duration_total",
    "fixation_mean", "fixation_std", "scan_frequency", "scan_amplitude",
    "scan_entropy", "dwell_cluster_variance", "yaw_skewness",
    "yaw_kurtosis", "pitch_variance", "roll_variance",
    "head_stability_index", "dominant_frequency", "spectral_entropy",
    "spectral_centroid", "spectral_spread", "band_power_low",
    "band_power_mid", "band_power_high", "trajectory_speed_mean",
    "trajectory_speed_std", "trajectory_curvature_mean",
    "trajectory_tortuosity", "head_displacement_magnitude",
    "vertical_scan_amplitude", "lateral_scan_amplitude",
    "forward_scan_amplitude", "scanpath_entropy",
}
RESPIRATION_AMBIGUOUS_SET = {
    "spectral_entropy_x",
    "spectral_centroid_x",
    "spectral_entropy_y",
    "spectral_centroid_y",
}


def assign_modality(column: Any) -> str:
    """Map a predictor to its acquisition modality using audited name rules."""
    name = str(column).casefold()
    if name in {"run", "flight_hours", "level"}:
        return "Context"
    if name in HEAD_MOVEMENT_SET:
        return "Head Movement"
    if name in RESPIRATION_AMBIGUOUS_SET:
        return "Respiration"
    if name.startswith("ppg_"):
        return "PPG"
    if name.startswith("hrv_"):
        return "ECG/HRV"
    if name.startswith(("scr_", "scl_", "phasic_")):
        return "EDA"
    if name.startswith(("flexor_", "extensor_")):
        return "EMG"
    if name.startswith(("accelerometry_forearm", "forearm_magnitude")):
        return "Forearm Accel."
    if name.startswith(("accelerometry_torso", "torso_magnitude")):
        return "Torso Accel."
    if name.startswith(
        (
            "overall_gaze_entropy", "psd_max", "psd_freq",
            "eyes_closed_fraction", "pupil_diam", "fix_", "sac_",
        )
    ):
        return "Eye Movement"
    if name.startswith(
        (
            "respiration_", "resp_", "inhalation_", "exhalation_", "average_tidal",
            "minute_ventilation", "duty_cycle", "peak_amplitude",
            "rms_respiration", "total_power", "total_wavelet_energy",
            "signal_entropy", "spectral_entropy", "spectral_centroid",
            "peak_frequency", "rmssd", "iqr_bbi", "skewness", "kurtosis",
            "average_peak_interval",
        )
    ):
        return "Respiration"
    return "Other"


def feature_modality_frame(features: list[Any], pd: Any) -> Any:
    result = pd.DataFrame(
        {
            "feature": [str(feature) for feature in features],
            "modality": [assign_modality(feature) for feature in features],
        }
    )
    result["modality"] = pd.Categorical(
        result["modality"], categories=MODALITY_ORDER, ordered=True
    )
    return result.sort_values(["modality", "feature"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark TabFM Ensemble, XGBoost, Random Forest, and KNN for "
            "difficulty classification and performance regression."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sheet", default="data")
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--level-column", default="level")
    parser.add_argument("--run-column", default="run")
    parser.add_argument("--flight-hours-column", default="flight_hours")
    parser.add_argument("--performance-column", default="performance")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_CHOICES,
        default=list(TASK_CHOICES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=list(MODEL_CHOICES),
    )
    parser.add_argument(
        "--feature-set",
        choices=("sensors_only", "sensors_plus_context"),
        default="sensors_only",
        help=(
            "sensors_only excludes the first five metadata/target columns; "
            "sensors_plus_context adds run/flight_hours and adds level only "
            "for regression."
        ),
    )
    parser.add_argument(
        "--split-protocol",
        choices=("paper_sample", "grouped_pilot"),
        default="paper_sample",
        help=(
            "paper_sample allows pilot overlap; grouped_pilot tests unseen "
            "pilots."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=3,
        help="Inner CV folds used only within each outer training fold.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-selector",
        choices=("mutual_info", "f_test"),
        default="mutual_info",
        help="Leakage-safe univariate selector fitted inside inner CV.",
    )
    parser.add_argument(
        "--feature-counts",
        type=int,
        nargs="+",
        default=[25, 50, 100, 150, 200],
        help=(
            "Candidate numbers of predictors. Values are clipped to the "
            "available training-fold columns; the best value is selected "
            "inside inner CV."
        ),
    )
    parser.add_argument(
        "--tuning-iterations",
        type=int,
        default=24,
        help="Randomized-search candidates per classical model and outer fold.",
    )
    parser.add_argument(
        "--selector-proxy-trees",
        type=int,
        default=250,
        help="Extra Trees estimators used to choose TabFM feature count.",
    )
    parser.add_argument(
        "--search-verbose",
        type=int,
        default=1,
        help="scikit-learn search progress verbosity.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32", "auto"),
        default="auto",
    )
    parser.add_argument("--tabfm-estimators", type=int, default=32)
    parser.add_argument("--tabfm-batch-size", type=int, default=1)
    parser.add_argument(
        "--tabfm-loader",
        choices=("low_memory", "official"),
        default="low_memory",
        help=(
            "low_memory streams safetensors directly into the final device "
            "and dtype, avoiding the official loader's large temporary RAM "
            "copies; official preserves TabFM's standard loading path."
        ),
    )
    parser.add_argument(
        "--progress-interval-minutes",
        type=float,
        default=5.0,
        help="Print a heartbeat during long TabFM fold computations.",
    )
    parser.add_argument("--xgb-trees", type=int, default=500)
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    parser.add_argument(
        "--explain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compute fold-local, model-agnostic SHAP permutation values. "
            "Enabled for all held-out runs by default in this release."
        ),
    )
    parser.add_argument(
        "--shap-method",
        choices=("permutation",),
        default="permutation",
        help=(
            "Direct fold-local permutation SHAP for the fitted model, using "
            "every ensemble member. Contributions are finite permutation "
            "estimates; surrogate and single-member modes are disabled."
        ),
    )
    parser.add_argument("--shap-background-size", type=int, default=12)
    parser.add_argument(
        "--shap-explain-rows",
        type=int,
        default=0,
        help=(
            "Held-out rows explained per fold; 0 explains every outer-test "
            "row (very expensive for TabFM)."
        ),
    )
    parser.add_argument(
        "--shap-max-evals",
        type=int,
        default=0,
        help="Permutation evaluations; 0 uses shap-permutations times (2F+1).",
    )
    parser.add_argument("--shap-permutations", type=int, default=10)
    parser.add_argument("--shap-batch-size", type=int, default=32)
    parser.add_argument(
        "--shap-checkpoint-rows",
        type=int,
        default=1,
        help=(
            "Held-out runs explained before atomically updating the fold "
            "SHAP CSV. A value of 1 provides run-level resume safety."
        ),
    )
    parser.add_argument(
        "--explain-models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=["tabfm_ensemble"],
        help="Models for which SHAP is computed when --explain is active.",
    )
    parser.add_argument(
        "--shap-source-rows",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional Excel source-row numbers to explain. Only their outer "
            "folds are refitted, enabling a fast screenshot/demo run."
        ),
    )
    parser.add_argument(
        "--shap-subject",
        type=str,
        default=None,
        help=(
            "Optional pilot identifier for targeted SHAP. Use together with "
            "--shap-runs; matching Excel source rows are resolved exactly."
        ),
    )
    parser.add_argument(
        "--shap-runs",
        nargs="+",
        type=int,
        default=None,
        help="Run numbers to explain for --shap-subject.",
    )
    parser.add_argument(
        "--shap-surrogate-trees",
        type=int,
        default=600,
        help="Extra Trees estimators per fold for --shap-method tree_surrogate.",
    )
    parser.add_argument(
        "--shap-surrogate-min-samples-leaf",
        type=int,
        default=2,
        help=(
            "Minimum leaf size for the cross-fitted SHAP surrogate. Small "
            "values improve local fidelity; values above 1 add smoothing."
        ),
    )
    parser.add_argument(
        "--shap-surrogate-min-r2",
        type=float,
        default=0.80,
        help=(
            "Minimum cross-fitted R-squared required before a fast-surrogate "
            "beeswarm is considered manuscript-ready."
        ),
    )
    parser.add_argument(
        "--allow-low-fidelity-shap",
        action="store_true",
        help=(
            "Export fast-surrogate figures even when cross-fitted R-squared "
            "is below --shap-surrogate-min-r2. The figure is visibly marked "
            "exploratory and should not be presented as TabFM SHAP."
        ),
    )
    parser.add_argument(
        "--shap-top-features",
        type=int,
        default=20,
        help="Features shown in SHAP beeswarm and mean-|SHAP| figures.",
    )
    parser.add_argument(
        "--shap-fast-members",
        type=int,
        default=1,
        help=(
            "TabFM members used by --shap-method fast_tabfm. One member is "
            "the fastest setting and is accepted only after fidelity audit."
        ),
    )
    parser.add_argument(
        "--shap-fast-rows-per-fold",
        type=int,
        default=6,
        help=(
            "Representative held-out rows explained per fold by fast_tabfm. "
            "All OOF rows are still used for the approximation-fidelity audit."
        ),
    )
    parser.add_argument(
        "--shap-fast-sampling",
        choices=("output_balanced", "experience_difficulty"),
        default="output_balanced",
        help=(
            "Representative-row design for fast_tabfm. output_balanced "
            "matches the original class/output-spanning sampler. "
            "experience_difficulty allocates equal rows within every fold to "
            "novice/easy, novice/hard, experienced/easy, and experienced/hard."
        ),
    )
    parser.add_argument(
        "--shap-fast-min-r2",
        type=float,
        default=0.90,
        help=(
            "Minimum OOF R-squared between the fast TabFM approximation and "
            "the saved full ensemble before manuscript figures are exported."
        ),
    )
    parser.add_argument("--figure-dpi", type=int, default=320)
    parser.add_argument(
        "--figure-formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png", "pdf", "svg"],
        help="Save each figure as high-resolution raster and/or vector output.",
    )
    parser.add_argument(
        "--download-missing-checkpoints",
        action="store_true",
        help=(
            "Allow TabFM to download a missing task checkpoint from Hugging "
            "Face. Each checkpoint is several GB."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Audit data, features, and folds without fitting any model.",
    )
    parser.add_argument(
        "--refresh-figures-only",
        action="store_true",
        help=(
            "Regenerate publication figures from existing result CSVs without "
            "loading or refitting models."
        ),
    )
    return parser.parse_args()


def resolve_column(columns: Any, requested: str) -> Any:
    if requested in columns:
        return requested
    matches = [
        column
        for column in columns
        if str(column).casefold() == str(requested).casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"Column {requested!r} was not found uniquely.")


def parse_sheet(value: Any) -> Any:
    text = str(value)
    return int(text) if text.isdigit() else value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def atomic_write_csv(frame: Any, path: Path) -> None:
    """Write a CSV in the destination directory and atomically replace it."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_tabfm_fold_state(estimator: Any, path: Path, joblib: Any) -> None:
    """Persist fitted TabFM context/weights without duplicating the backbone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    backbone = estimator.model
    try:
        estimator.model = None
        joblib.dump(estimator, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        estimator.model = backbone
        if temporary.exists():
            temporary.unlink()


def load_tabfm_fold_state(path: Path, backbone: Any, joblib: Any) -> Any:
    """Restore a lightweight fitted TabFM fold state onto one loaded backbone."""
    estimator = joblib.load(path)
    estimator.model = backbone
    return estimator


def resolve_targeted_shap_rows(
    args: argparse.Namespace,
    frame: Any,
    subject_column: Any,
    run_column: Any,
    source_excel_rows: Any,
    pd: Any,
    np: Any,
) -> None:
    """Resolve a pilot plus run numbers to unique Excel source rows."""
    if args.shap_subject is None:
        return
    requested_subject = str(args.shap_subject).strip()
    subject_text = frame[subject_column].astype(str).str.strip()
    subject_mask = subject_text.eq(requested_subject).to_numpy(dtype=bool)
    if not subject_mask.any():
        try:
            subject_number = float(requested_subject)
        except ValueError:
            subject_number = float("nan")
        numeric_subjects = pd.to_numeric(
            frame[subject_column], errors="coerce"
        ).to_numpy(dtype=float)
        if np.isfinite(subject_number):
            subject_mask = np.isclose(
                numeric_subjects, subject_number, equal_nan=False
            )
    if not subject_mask.any():
        available = sorted(
            frame[subject_column].dropna().astype(str).unique().tolist()
        )
        raise ValueError(
            f"--shap-subject {requested_subject!r} was not found. "
            f"Available subjects include: {available[:20]}"
        )
    run_values = pd.to_numeric(frame[run_column], errors="raise").to_numpy(
        dtype=int
    )
    resolved: list[int] = []
    for requested_run in args.shap_runs:
        matches = np.flatnonzero(
            subject_mask & (run_values == int(requested_run))
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one row for pilot {requested_subject}, "
                f"run {requested_run}; found {len(matches)}."
            )
        resolved.append(int(source_excel_rows[int(matches[0])]))
    args.shap_source_rows = resolved
    mapping = ", ".join(
        f"run {run}=Excel row {row}"
        for run, row in zip(args.shap_runs, resolved)
    )
    print(
        f"Targeted SHAP: pilot {requested_subject}; {mapping}.",
        flush=True,
    )


def validate_arguments(args: argparse.Namespace) -> None:
    if len(args.tasks) != len(set(args.tasks)):
        raise ValueError("--tasks contains duplicates.")
    if len(args.models) != len(set(args.models)):
        raise ValueError("--models contains duplicates.")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if args.inner_folds < 2:
        raise ValueError("--inner-folds must be at least 2.")
    if args.inner_folds >= args.folds and args.split_protocol == "grouped_pilot":
        print(
            "WARNING: inner folds are not required to be fewer than outer "
            "folds, but verify that each outer training fold has enough pilots."
        )
    if not args.feature_counts or any(value < 1 for value in args.feature_counts):
        raise ValueError("--feature-counts must contain positive integers.")
    if len(args.feature_counts) != len(set(args.feature_counts)):
        raise ValueError("--feature-counts contains duplicates.")
    if args.tuning_iterations < 1:
        raise ValueError("--tuning-iterations must be at least 1.")
    if args.selector_proxy_trees < 1:
        raise ValueError("--selector-proxy-trees must be at least 1.")
    if args.search_verbose < 0:
        raise ValueError("--search-verbose cannot be negative.")
    for name in (
        "tabfm_estimators",
        "tabfm_batch_size",
        "xgb_trees",
        "rf_trees",
        "knn_neighbors",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1.")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100.")
    if args.progress_interval_minutes <= 0:
        raise ValueError("--progress-interval-minutes must be greater than 0.")
    if not 0 < args.confidence_level < 1:
        raise ValueError("--confidence-level must be between 0 and 1.")
    for name in (
        "shap_background_size",
        "shap_batch_size",
        "shap_checkpoint_rows",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1.")
    if args.shap_explain_rows < 0:
        raise ValueError("--shap-explain-rows cannot be negative.")
    if args.shap_permutations < 1:
        raise ValueError("--shap-permutations must be positive.")
    if args.shap_max_evals < 0:
        raise ValueError("--shap-max-evals cannot be negative.")
    if args.shap_surrogate_trees < 50:
        raise ValueError("--shap-surrogate-trees must be at least 50.")
    if args.shap_surrogate_min_samples_leaf < 1:
        raise ValueError(
            "--shap-surrogate-min-samples-leaf must be at least 1."
        )
    if not -1.0 <= args.shap_surrogate_min_r2 <= 1.0:
        raise ValueError("--shap-surrogate-min-r2 must be between -1 and 1.")
    if args.shap_top_features < 5:
        raise ValueError("--shap-top-features must be at least 5.")
    if args.shap_fast_members < 1:
        raise ValueError("--shap-fast-members must be at least 1.")
    if args.shap_fast_rows_per_fold < 2:
        raise ValueError("--shap-fast-rows-per-fold must be at least 2.")
    if (
        args.shap_fast_sampling == "experience_difficulty"
        and args.shap_fast_rows_per_fold % 4 != 0
    ):
        raise ValueError(
            "--shap-fast-rows-per-fold must be divisible by 4 when "
            "--shap-fast-sampling experience_difficulty is used."
        )
    if not -1.0 <= args.shap_fast_min_r2 <= 1.0:
        raise ValueError("--shap-fast-min-r2 must be between -1 and 1.")
    if args.shap_source_rows is not None:
        if any(value < 2 for value in args.shap_source_rows):
            raise ValueError("--shap-source-rows must contain Excel rows >= 2.")
        if len(args.shap_source_rows) != len(set(args.shap_source_rows)):
            raise ValueError("--shap-source-rows contains duplicates.")
    if (args.shap_subject is None) != (args.shap_runs is None):
        raise ValueError(
            "Use --shap-subject and --shap-runs together."
        )
    if args.shap_source_rows is not None and args.shap_subject is not None:
        raise ValueError(
            "Choose either --shap-source-rows or --shap-subject/--shap-runs."
        )
    if args.shap_runs is not None:
        if any(value < 1 for value in args.shap_runs):
            raise ValueError("--shap-runs must contain positive run numbers.")
        if len(args.shap_runs) != len(set(args.shap_runs)):
            raise ValueError("--shap-runs contains duplicates.")
    if (
        args.shap_source_rows is not None or args.shap_subject is not None
    ) and not args.explain:
        raise ValueError(
            "Targeted SHAP arguments require --explain."
        )
    if args.explain and not set(args.explain_models).issubset(args.models):
        raise ValueError("--explain-models must be a subset of --models.")
    if args.shap_method in {"fast_tabfm", "tree_surrogate"}:
        if not args.explain:
            raise ValueError(
                f"--shap-method {args.shap_method} requires --explain."
            )
        if set(args.explain_models) != {"tabfm_ensemble"}:
            raise ValueError(
                f"--shap-method {args.shap_method} currently explains only "
                "--explain-models tabfm_ensemble."
            )
        if args.shap_source_rows is not None or args.shap_subject is not None:
            raise ValueError(
                "Targeted row arguments apply to direct permutation SHAP; "
                "fast_surrogate always explains all saved OOF rows."
            )
    if args.figure_dpi < 200:
        raise ValueError("--figure-dpi must be at least 200 for publication output.")
    if len(args.figure_formats) != len(set(args.figure_formats)):
        raise ValueError("--figure-formats contains duplicates.")
    if args.validate_only and args.refresh_figures_only:
        raise ValueError("Choose either --validate-only or --refresh-figures-only.")


def resolve_device_dtype(args: argparse.Namespace, torch: Any) -> tuple[str, Any]:
    from workflow_paths import resolve_runtime
    return resolve_runtime(torch, args.device, args.dtype)


def checkpoint_argument(
    checkpoint_root: Path,
    task: str,
    allow_download: bool,
) -> str | None:
    task_dir = checkpoint_root / task
    if task_dir.is_dir() and (task_dir / "model.safetensors").is_file():
        return str(checkpoint_root)
    if allow_download:
        print(
            f"WARNING: local TabFM {task} checkpoint is missing; downloading "
            "several GB from Hugging Face."
        )
        return None
    raise FileNotFoundError(
        f"The local TabFM {task} checkpoint is missing at {task_dir}. "
        f"Run python download_checkpoints.py --task {task} with the same "
        "checkpoint_root configuration, or download the official checkpoint "
        "to the --checkpoint-path directory. See README.md."
    )


def load_tabfm_low_memory(
    model_type: str,
    checkpoint_path: str | None,
    device: str,
    dtype: Any,
    torch: Any,
) -> Any:
    """Load a TabFM safetensors checkpoint without a full fp32 RAM copy.

    TabFM's official Hugging Face loader first materializes the complete
    float32 checkpoint on the CPU and subsequently casts/moves the model. On
    memory-constrained Windows systems, that can require well over twice the
    checkpoint size and lead to paging or a native torch crash. This loader
    constructs the module on PyTorch's meta device and streams every tensor
    directly into its final device and dtype before assigning it to the model.
    """
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from tabfm.src.pytorch.tabfm_v1_0_0 import TabFM_HF
    except ImportError as exc:
        raise RuntimeError(
            "The low-memory TabFM loader requires huggingface_hub and "
            "safetensors. Install requirements.txt in your active environment."
        ) from exc

    if model_type not in ("classification", "regression"):
        raise ValueError(f"Unsupported TabFM model type: {model_type!r}")

    if checkpoint_path is None:
        print(
            f"Downloading the missing TabFM {model_type} checkpoint...",
            flush=True,
        )
        checkpoint_root = Path(
            snapshot_download(
                repo_id="google/tabfm-1.0.0-pytorch",
                allow_patterns=[f"{model_type}/**"],
            )
        )
    else:
        checkpoint_root = Path(checkpoint_path).expanduser().resolve()

    task_dir = checkpoint_root
    nested_task_dir = checkpoint_root / model_type
    if nested_task_dir.is_dir():
        task_dir = nested_task_dir
    config_path = task_dir / "config.json"
    weights_path = task_dir / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"Incomplete TabFM {model_type} checkpoint in {task_dir}; "
            "config.json and model.safetensors are both required."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "is_classifier" not in config and "task" in config:
        config["is_classifier"] = config.pop("task") == "classification"
    for metadata_key in ("model_type", "version", "framework"):
        config.pop(metadata_key, None)
    config.setdefault("is_classifier", model_type == "classification")
    expected_classifier = model_type == "classification"
    if bool(config["is_classifier"]) != expected_classifier:
        raise ValueError(
            f"Checkpoint configuration at {config_path} does not match "
            f"the requested {model_type} task."
        )

    print(
        f"Low-memory loader: constructing {model_type} model on meta device...",
        flush=True,
    )
    with torch.device("meta"):
        model = TabFM_HF(**config)

    state: dict[str, Any] = {}
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            tensor_names = list(handle.keys())
            print(
                "Low-memory loader: streaming "
                f"{len(tensor_names)} tensors to {device} ({dtype})...",
                flush=True,
            )
            for tensor_number, name in enumerate(tensor_names, start=1):
                source_tensor = handle.get_tensor(name)
                if dtype is not None and source_tensor.is_floating_point():
                    target_tensor = source_tensor.to(device=device, dtype=dtype)
                else:
                    target_tensor = source_tensor.to(device=device)
                state[name] = target_tensor
                del source_tensor, target_tensor
                if tensor_number % 100 == 0 or tensor_number == len(tensor_names):
                    progress = (
                        f"  loaded {tensor_number}/{len(tensor_names)} tensors"
                    )
                    if device == "cuda":
                        allocated_gb = torch.cuda.memory_allocated() / 1024**3
                        progress += f" | GPU allocated={allocated_gb:.2f} GB"
                    print(progress, flush=True)

        model.load_state_dict(state, strict=True, assign=True)
        state.clear()
        gc.collect()
        model.eval()
        meta_tensors = [
            name
            for name, tensor in list(model.named_parameters())
            + list(model.named_buffers())
            if tensor.device.type == "meta"
        ]
        if meta_tensors:
            raise RuntimeError(
                "Low-memory loading left tensors on the meta device: "
                f"{meta_tensors[:10]}"
            )
        print("Low-memory TabFM checkpoint loaded successfully.", flush=True)
        return model
    except Exception:
        state.clear()
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        raise


@contextmanager
def progress_heartbeat(label: str, interval_minutes: float) -> Any:
    """Periodically report that a long, blocking model call is still alive."""
    started = time.monotonic()
    stopped = threading.Event()

    def report() -> None:
        interval_seconds = interval_minutes * 60
        while not stopped.wait(interval_seconds):
            elapsed_minutes = (time.monotonic() - started) / 60
            print(
                f"  still running {label} | elapsed={elapsed_minutes:.1f} min",
                flush=True,
            )

    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()
    print(f"  starting {label}...", flush=True)
    try:
        yield
    finally:
        stopped.set()
        reporter.join(timeout=1)
        elapsed_minutes = (time.monotonic() - started) / 60
        print(
            f"  finished {label} | elapsed={elapsed_minutes:.1f} min",
            flush=True,
        )


def classification_metrics(
    y_true: Any,
    predictions: Any,
    probabilities: Any,
    metric_functions: dict[str, Any],
    np: Any,
) -> dict[str, float]:
    classes = np.asarray(["easy", "hard"])
    hard_truth = (np.asarray(y_true) == "hard").astype(float)
    hard_probability = np.asarray(probabilities[:, 1], dtype=float)
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_index = np.minimum(np.digitize(hard_probability, bin_edges) - 1, 9)
    ece = 0.0
    for index in range(10):
        mask = bin_index == index
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(hard_probability[mask]))
                - float(np.mean(hard_truth[mask]))
            )
    return {
        "accuracy": float(
            metric_functions["accuracy_score"](y_true, predictions)
        ),
        "balanced_accuracy": float(
            metric_functions["balanced_accuracy_score"](
                y_true,
                predictions,
            )
        ),
        "f1_macro": float(
            metric_functions["f1_score"](
                y_true,
                predictions,
                average="macro",
            )
        ),
        "roc_auc_hard": float(
            metric_functions["roc_auc_score"](
                hard_truth,
                hard_probability,
            )
        ),
        "log_loss": float(
            metric_functions["log_loss"](
                y_true,
                probabilities,
                labels=classes,
            )
        ),
        "brier_score": float(np.mean((hard_probability - hard_truth) ** 2)),
        "ece_10bin": float(ece),
    }


def regression_metrics(
    y_true: Any,
    predictions: Any,
    metric_functions: dict[str, Any],
    np: Any,
) -> dict[str, float]:
    y_array = np.asarray(y_true, dtype=float)
    prediction_array = np.asarray(predictions, dtype=float)
    return {
        "mae": float(
            metric_functions["mean_absolute_error"](
                y_array,
                prediction_array,
            )
        ),
        "rmse": float(
            np.sqrt(
                metric_functions["mean_squared_error"](
                    y_array,
                    prediction_array,
                )
            )
        ),
        "r2": float(
            metric_functions["r2_score"](y_array, prediction_array)
        ),
        "median_absolute_error": float(
            metric_functions["median_absolute_error"](
                y_array,
                prediction_array,
            )
        ),
        "mape_percent": float(
            metric_functions["mean_absolute_percentage_error"](
                y_array,
                prediction_array,
            )
            * 100.0
        ),
    }


def compute_metrics(
    task: str,
    predictions_frame: Any,
    metric_functions: dict[str, Any],
    np: Any,
) -> dict[str, float]:
    if task == "classification":
        return classification_metrics(
            predictions_frame["actual_class"].to_numpy(),
            predictions_frame["predicted_class"].to_numpy(),
            predictions_frame[
                ["probability_easy", "probability_hard"]
            ].to_numpy(dtype=float),
            metric_functions,
            np,
        )
    return regression_metrics(
        predictions_frame["actual_performance"].to_numpy(dtype=float),
        predictions_frame["predicted_performance"].to_numpy(dtype=float),
        metric_functions,
        np,
    )


def regression_strata(y: Any, folds: int, pd: Any) -> Any:
    """Return the largest quantile stratification with >= folds rows/bin."""
    values = pd.Series(y).rank(method="first")
    max_bins = min(10, max(2, len(values) // folds))
    for bins in range(max_bins, 1, -1):
        codes = pd.qcut(values, q=bins, labels=False, duplicates="drop")
        if int(codes.value_counts().min()) >= folds:
            return codes.to_numpy(dtype=int)
    raise ValueError("Performance cannot be stratified into at least two bins.")


def make_splits(
    task: str,
    protocol: str,
    levels: Any,
    performance: Any,
    subjects: Any,
    folds: int,
    seed: int,
    splitters: dict[str, Any],
    pd: Any,
    np: Any,
) -> list[tuple[int, Any, Any]]:
    row_indices = np.arange(len(levels), dtype=int)
    if protocol == "paper_sample":
        if task == "classification":
            strata = levels
        else:
            strata = regression_strata(performance, folds, pd)
        splitter = splitters["StratifiedKFold"](
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        )
        raw_splits = splitter.split(row_indices, strata)
    elif task == "classification":
        splitter = splitters["StratifiedGroupKFold"](
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        )
        raw_splits = splitter.split(row_indices, levels, groups=subjects)
    else:
        splitter = splitters["GroupKFold"](n_splits=folds)
        raw_splits = splitter.split(row_indices, performance, groups=subjects)
    return [
        (fold, train_indices, test_indices)
        for fold, (train_indices, test_indices) in enumerate(
            raw_splits,
            start=1,
        )
    ]


def candidate_feature_counts(requested: list[int], available: int) -> list[int]:
    """Return unique, valid feature-count candidates for a training fold."""
    if available < 1:
        raise ValueError("Feature selection received no usable predictors.")
    counts = sorted({min(int(value), available) for value in requested})
    if not counts:
        raise ValueError("No valid feature-count candidates remain.")
    return counts


def make_inner_cv(
    task: str,
    protocol: str,
    levels: Any,
    performance: Any,
    subjects: Any,
    folds: int,
    seed: int,
    splitters: dict[str, Any],
    pd: Any,
    np: Any,
) -> list[tuple[Any, Any]]:
    """Create inner splits using only the current outer training rows."""
    nested = make_splits(
        task=task,
        protocol=protocol,
        levels=levels,
        performance=performance,
        subjects=subjects,
        folds=folds,
        seed=seed,
        splitters=splitters,
        pd=pd,
        np=np,
    )
    return [(train_indices, validation_indices) for _, train_indices, validation_indices in nested]


def make_selector_score_function(
    task: str,
    selector_name: str,
    seed: int,
    mutual_info_classif: Any,
    mutual_info_regression: Any,
    f_classif: Any,
    f_regression: Any,
) -> Any:
    if selector_name == "f_test":
        return f_classif if task == "classification" else f_regression
    function = (
        mutual_info_classif
        if task == "classification"
        else mutual_info_regression
    )
    return partial(function, random_state=seed)


def search_scoring(task: str) -> str:
    return "roc_auc" if task == "classification" else "neg_root_mean_squared_error"


def encoded_search_target(task: str, y: Any, np: Any) -> Any:
    if task == "classification":
        return (np.asarray(y) == "hard").astype(int)
    return np.asarray(y, dtype=float)


def build_classical_search(
    model_key: str,
    task: str,
    feature_counts: list[int],
    inner_cv: list[tuple[Any, Any]],
    args: argparse.Namespace,
    score_function: Any,
    classes: dict[str, Any],
) -> Any:
    """Build a leakage-safe randomized search for a classical model."""
    Pipeline = classes["Pipeline"]
    SelectKBest = classes["SelectKBest"]
    SimpleImputer = classes["SimpleImputer"]
    StandardScaler = classes["StandardScaler"]
    RandomizedSearchCV = classes["RandomizedSearchCV"]

    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(score_func=score_function)),
    ]
    parameter_distributions: dict[str, Any] = {
        "selector__k": feature_counts,
    }
    if model_key == "xgboost":
        if task == "classification":
            estimator = classes["XGBClassifier"](
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=args.seed,
                n_jobs=1,
            )
        else:
            estimator = classes["XGBRegressor"](
                objective="reg:squarederror",
                tree_method="hist",
                random_state=args.seed,
                n_jobs=1,
            )
        parameter_distributions.update(
            {
                "model__n_estimators": sorted({200, args.xgb_trees, 800, 1200}),
                "model__max_depth": [2, 3, 4, 6],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__min_child_weight": [1, 3, 5, 10],
                "model__subsample": [0.65, 0.8, 1.0],
                "model__colsample_bytree": [0.5, 0.7, 0.9, 1.0],
                "model__reg_alpha": [0.0, 0.01, 0.1, 1.0],
                "model__reg_lambda": [0.5, 1.0, 5.0, 10.0],
            }
        )
    elif model_key == "random_forest":
        if task == "classification":
            estimator = classes["RandomForestClassifier"](
                class_weight="balanced_subsample",
                random_state=args.seed,
                n_jobs=1,
            )
        else:
            estimator = classes["RandomForestRegressor"](
                random_state=args.seed,
                n_jobs=1,
            )
        parameter_distributions.update(
            {
                "model__n_estimators": sorted({300, args.rf_trees, 800}),
                "model__max_depth": [None, 8, 16, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", 0.2, 0.5, 0.8],
                "model__max_samples": [None, 0.7, 0.9],
            }
        )
    elif model_key == "knn":
        steps.append(("scaler", StandardScaler()))
        estimator_class = (
            classes["KNeighborsClassifier"]
            if task == "classification"
            else classes["KNeighborsRegressor"]
        )
        estimator = estimator_class(n_jobs=1)
        parameter_distributions.update(
            {
                "model__n_neighbors": sorted(
                    {3, 5, 7, 11, 15, 21, 31, args.knn_neighbors}
                ),
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
                "model__leaf_size": [20, 30, 50],
            }
        )
    else:
        raise ValueError(f"Unsupported optimized classical model: {model_key}")

    steps.append(("model", estimator))
    pipeline = Pipeline(steps)
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=args.tuning_iterations,
        scoring=search_scoring(task),
        cv=inner_cv,
        refit=True,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=args.search_verbose,
        error_score="raise",
        return_train_score=False,
    )


def build_tabfm_feature_search(
    task: str,
    feature_counts: list[int],
    inner_cv: list[tuple[Any, Any]],
    args: argparse.Namespace,
    score_function: Any,
    classes: dict[str, Any],
) -> Any:
    """Choose TabFM feature count with a fast, nested Extra Trees proxy."""
    if task == "classification":
        proxy = classes["ExtraTreesClassifier"](
            n_estimators=args.selector_proxy_trees,
            min_samples_leaf=2,
            class_weight="balanced",
            max_features="sqrt",
            random_state=args.seed,
            n_jobs=1,
        )
    else:
        proxy = classes["ExtraTreesRegressor"](
            n_estimators=args.selector_proxy_trees,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=args.seed,
            n_jobs=1,
        )
    pipeline = classes["Pipeline"](
        [
            ("imputer", classes["SimpleImputer"](strategy="median")),
            ("selector", classes["SelectKBest"](score_func=score_function)),
            ("proxy", proxy),
        ]
    )
    return classes["GridSearchCV"](
        estimator=pipeline,
        param_grid={"selector__k": feature_counts},
        scoring=search_scoring(task),
        cv=inner_cv,
        refit=True,
        n_jobs=args.n_jobs,
        verbose=args.search_verbose,
        error_score="raise",
        return_train_score=False,
    )


def feature_audit_from_pipeline(
    pipeline: Any,
    feature_names: list[Any],
    pd: Any,
    np: Any,
) -> Any:
    """Return per-feature selection, score, imputation, and scaling details."""
    selector = pipeline.named_steps["selector"]
    support = np.asarray(selector.get_support(), dtype=bool)
    scores = np.asarray(selector.scores_, dtype=float)
    safe_scores = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-safe_scores, kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    audit = pd.DataFrame(
        {
            "feature": [str(name) for name in feature_names],
            "modality": [assign_modality(name) for name in feature_names],
            "selected": support,
            "selection_score": scores,
            "selection_rank": ranks,
            "training_imputation_median": pipeline.named_steps[
                "imputer"
            ].statistics_,
        }
    )
    scaler = pipeline.named_steps.get("scaler")
    if scaler is not None:
        audit["training_scaler_mean"] = np.nan
        audit["training_scaler_scale"] = np.nan
        audit.loc[support, "training_scaler_mean"] = scaler.mean_
        audit.loc[support, "training_scaler_scale"] = scaler.scale_
    return audit.sort_values(
        ["selected", "selection_rank"],
        ascending=[False, True],
    )


def compact_search_results(search: Any, pd: Any) -> Any:
    """Keep the useful, portable columns from sklearn CV results."""
    results = pd.DataFrame(search.cv_results_)
    columns = [
        column
        for column in results.columns
        if column.startswith("param_")
        or column
        in {
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
        }
    ]
    return results[columns].sort_values("rank_test_score")


def cluster_bootstrap(
    task: str,
    predictions_frame: Any,
    point_metrics: dict[str, float],
    metric_functions: dict[str, Any],
    n_resamples: int,
    confidence_level: float,
    seed: int,
    pd: Any,
    np: Any,
) -> tuple[Any, Any]:
    pilot_groups = {
        pilot: np.asarray(indices, dtype=int)
        for pilot, indices in predictions_frame.groupby(
            "subject",
            sort=True,
        ).indices.items()
    }
    pilots = np.asarray(list(pilot_groups), dtype=object)
    if len(pilots) < 2:
        raise ValueError("At least two pilots are required for bootstrap.")
    rng = np.random.default_rng(seed)
    draw_records: list[dict[str, Any]] = []
    for resample in range(1, n_resamples + 1):
        sampled_pilots = rng.choice(pilots, size=len(pilots), replace=True)
        sampled_rows = np.concatenate(
            [pilot_groups[pilot] for pilot in sampled_pilots]
        )
        sampled = predictions_frame.iloc[sampled_rows]
        record: dict[str, Any] = {
            "bootstrap_resample": resample,
            "n_rows": int(len(sampled)),
            "n_unique_pilots": int(len(set(sampled_pilots.tolist()))),
        }
        record.update(compute_metrics(task, sampled, metric_functions, np))
        draw_records.append(record)
    draws = pd.DataFrame(draw_records)
    alpha = (1.0 - confidence_level) / 2.0
    summaries: list[dict[str, Any]] = []
    for metric_name, point_estimate in point_metrics.items():
        if metric_name not in draws:
            continue
        values = draws[metric_name].dropna().to_numpy(dtype=float)
        summaries.append(
            {
                "metric": metric_name,
                "point_estimate": float(point_estimate),
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_standard_error": float(np.std(values, ddof=1)),
                "ci_lower": float(np.quantile(values, alpha)),
                "ci_upper": float(np.quantile(values, 1.0 - alpha)),
                "confidence_level": float(confidence_level),
                "n_valid_resamples": int(len(values)),
                "resampling_unit": "pilot",
            }
        )
    return pd.DataFrame(summaries), draws


def paired_differences(
    task: str,
    metrics_by_model: dict[str, dict[str, float]],
    draws_by_model: dict[str, Any],
    confidence_level: float,
    pd: Any,
    np: Any,
) -> Any:
    reference = "tabfm_ensemble"
    if reference not in draws_by_model:
        return pd.DataFrame()
    higher_is_better = (
        {"accuracy", "balanced_accuracy", "f1_macro", "roc_auc_hard"}
        if task == "classification"
        else {"r2"}
    )
    alpha = (1.0 - confidence_level) / 2.0
    records: list[dict[str, Any]] = []
    reference_draws = draws_by_model[reference].sort_values(
        "bootstrap_resample"
    )
    for comparator, comparator_draws_raw in draws_by_model.items():
        if comparator == reference:
            continue
        comparator_draws = comparator_draws_raw.sort_values(
            "bootstrap_resample"
        )
        for metric_name in metrics_by_model[reference]:
            if metric_name not in comparator_draws:
                continue
            if metric_name in higher_is_better:
                differences = (
                    reference_draws[metric_name].to_numpy(dtype=float)
                    - comparator_draws[metric_name].to_numpy(dtype=float)
                )
                point = (
                    metrics_by_model[reference][metric_name]
                    - metrics_by_model[comparator][metric_name]
                )
                definition = "tabfm_minus_comparator"
            else:
                differences = (
                    comparator_draws[metric_name].to_numpy(dtype=float)
                    - reference_draws[metric_name].to_numpy(dtype=float)
                )
                point = (
                    metrics_by_model[comparator][metric_name]
                    - metrics_by_model[reference][metric_name]
                )
                definition = "comparator_minus_tabfm"
            valid = differences[np.isfinite(differences)]
            records.append(
                {
                    "reference_model": reference,
                    "comparator_model": comparator,
                    "metric": metric_name,
                    "difference_definition": (
                        f"{definition}; positive_favors_tabfm"
                    ),
                    "point_advantage_tabfm": float(point),
                    "ci_lower": float(np.quantile(valid, alpha)),
                    "ci_upper": float(np.quantile(valid, 1.0 - alpha)),
                    "probability_tabfm_better": float(np.mean(valid > 0)),
                    "n_valid_resamples": int(len(valid)),
                }
            )
    return pd.DataFrame(records)


def align_binary_probabilities(
    raw_probabilities: Any,
    model_classes: Any,
    np: Any,
) -> tuple[Any, float]:
    target_classes = np.asarray(["easy", "hard"])
    raw = np.asarray(raw_probabilities, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all():
        raise RuntimeError("Classifier returned invalid binary probabilities.")
    aligned = np.zeros_like(raw, dtype=float)
    for target_position, target_class in enumerate(target_classes):
        matches = np.where(np.asarray(model_classes) == target_class)[0]
        if len(matches) != 1:
            raise RuntimeError(
                f"Classifier probability output lacks class {target_class!r}."
            )
        aligned[:, target_position] = raw[:, int(matches[0])]
    deviation = float(np.max(np.abs(aligned.sum(axis=1) - 1.0)))
    aligned = np.clip(aligned, 0.0, None)
    row_sums = aligned.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise RuntimeError("Classifier returned a zero-probability row.")
    return aligned / row_sums, deviation


def feature_columns_for_task(
    task: str,
    feature_set: str,
    columns: Any,
    subject_column: Any,
    level_column: Any,
    run_column: Any,
    flight_hours_column: Any,
    performance_column: Any,
) -> list[Any]:
    metadata = {
        subject_column,
        level_column,
        run_column,
        flight_hours_column,
        performance_column,
    }
    features = [column for column in columns if column not in metadata]
    if feature_set == "sensors_plus_context":
        features.extend([run_column, flight_hours_column])
        if task == "regression":
            features.append(level_column)
    return features


def common_prediction_columns(
    frame: Any,
    indices: Any,
    source_excel_rows: Any,
    model_key: str,
    fold_number: int,
    subject_column: Any,
    level_column: Any,
    run_column: Any,
    flight_hours_column: Any,
    pd: Any,
) -> dict[str, Any]:
    return {
        "model": model_key,
        "source_excel_row": source_excel_rows[indices],
        "fold": fold_number,
        "subject": frame.iloc[indices][subject_column].to_numpy(),
        "level": frame.iloc[indices][level_column].to_numpy(),
        "run": frame.iloc[indices][run_column].to_numpy(),
        "flight_hours": frame.iloc[indices][flight_hours_column].to_numpy(),
    }


def validate_saved_fold(
    path: Path,
    model_key: str,
    fold_number: int,
    expected_rows: set[int],
    pd: Any,
) -> bool:
    if not path.exists():
        return False
    saved = pd.read_csv(path)
    actual_rows = set(saved["source_excel_row"].astype(int).tolist())
    if (
        len(saved) != len(expected_rows)
        or actual_rows != expected_rows
        or set(saved["fold"].astype(int)) != {fold_number}
        or set(saved["model"].astype(str)) != {model_key}
    ):
        raise RuntimeError(
            f"Existing resumable fold is incompatible: {path}. "
            "Choose a new --output-dir or remove only this invalid fold."
        )
    return True


def validate_saved_shap(path, model_key, fold_number, expected_rows, args, pd):
    from direct_shap import meta_path, validate_group
    if not path.is_file() or not meta_path(path).is_file():
        return False
    metadata = json.loads(meta_path(path).read_text(encoding="utf-8"))
    if metadata["seed"] != args.seed or metadata["fold"] != fold_number:
        return False
    if metadata["max_evals"] != (args.shap_max_evals or (2 * len(metadata["features"]) + 1) * args.shap_permutations):
        return False
    if len(metadata["background_positions"]) != args.shap_background_size:
        return False
    frame = pd.read_csv(path)
    for _, group in frame.groupby("source_excel_row"):
        validate_group(group, metadata)
    explained = set(frame.source_excel_row.astype(int))
    wanted = expected_rows.intersection(args.shap_source_rows) if args.shap_source_rows else expected_rows
    return wanted <= explained if args.shap_explain_rows == 0 else len(wanted & explained) >= min(args.shap_explain_rows, len(wanted))


def validate_oof(
    predictions: Any,
    source_excel_rows: Any,
    model_key: str,
) -> None:
    expected = set(source_excel_rows.tolist())
    if (
        len(predictions) != len(source_excel_rows)
        or predictions["source_excel_row"].nunique() != len(source_excel_rows)
        or set(predictions["source_excel_row"].astype(int)) != expected
        or set(predictions["model"].astype(str)) != {model_key}
    ):
        raise RuntimeError(
            f"{MODEL_DISPLAY[model_key]} out-of-fold predictions do not cover "
            "every source row exactly once."
        )


def regression_target_summary(performance: Any, np: Any) -> dict[str, float]:
    values = np.asarray(performance, dtype=float)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
    }


def format_ci_payload(stability: Any) -> dict[str, Any]:
    return {
        str(row["metric"]): {
            "lower": float(row["ci_lower"]),
            "upper": float(row["ci_upper"]),
            "bootstrap_standard_error": float(
                row["bootstrap_standard_error"]
            ),
        }
        for row in stability.to_dict(orient="records")
    }


def configure_publication_style(plt: Any) -> None:
    """Apply one journal-ready, colorblind-safe visual language."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(
    figure: Any,
    output_stem: Path,
    formats: list[str],
    dpi: int,
    plt: Any,
) -> None:
    """Save a figure in requested raster/vector formats, then release it."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        figure.savefig(
            output_stem.with_suffix(f".{suffix}"),
            dpi=dpi if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def plot_task_comparison(
    task: str,
    comparison: Any,
    task_dir: Path,
    plt: Any,
    np: Any,
    figure_formats: list[str],
    figure_dpi: int,
) -> None:
    """Forest-style model comparison with pilot-bootstrap confidence intervals."""
    if task == "classification":
        metrics = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_hard"]
        labels = ["Accuracy", "Balanced accuracy", "Macro F1", "ROC-AUC"]
    else:
        metrics = ["mae", "rmse", "r2"]
        labels = ["MAE ↓", "RMSE ↓", "R² ↑"]
    figure, axes = plt.subplots(
        1, len(metrics), figsize=(4.0 * len(metrics), 4.8), squeeze=False
    )
    ordered = comparison.sort_values("rank").reset_index(drop=True)
    y_positions = np.arange(len(ordered))[::-1]
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        axis = axes[0, index]
        values = ordered[metric].to_numpy(dtype=float)
        lower = ordered[f"{metric}_ci_lower"].to_numpy(dtype=float)
        upper = ordered[f"{metric}_ci_upper"].to_numpy(dtype=float)
        for position, row_index in enumerate(range(len(ordered))):
            model_key = str(ordered.loc[row_index, "model"])
            axis.errorbar(
                values[row_index],
                y_positions[position],
                xerr=np.asarray(
                    [[max(0.0, values[row_index] - lower[row_index])],
                     [max(0.0, upper[row_index] - values[row_index])]]
                ),
                fmt="o",
                color=MODEL_COLORS.get(model_key, "#555555"),
                markersize=7,
                elinewidth=2,
                capsize=3,
                zorder=3,
            )
        axis.set_title(label)
        axis.grid(axis="x")
        axis.set_yticks(y_positions)
        if index == 0:
            axis.set_yticklabels(ordered["model_display_name"])
        else:
            axis.set_yticklabels([])
        if task == "classification":
            visible_min = max(0.0, float(np.nanmin(lower)) - 0.05)
            axis.set_xlim(visible_min, 1.01)
        if metric == "r2":
            axis.axvline(0.0, color="#555555", linewidth=0.9, linestyle="--")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    protocol = "same outer folds; 95% pilot-cluster bootstrap CIs"
    figure.suptitle(f"{TASK_DISPLAY[task]} — {protocol}", fontsize=14, y=1.03)
    figure.tight_layout()
    save_figure(
        figure, task_dir / "figure_model_comparison", figure_formats,
        figure_dpi, plt
    )


def plot_classification_diagnostics(
    predictions_by_model: dict[str, Any],
    task_dir: Path,
    metric_functions: dict[str, Any],
    plt: Any,
    np: Any,
    figure_formats: list[str],
    figure_dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    for model_key, predictions in predictions_by_model.items():
        truth = predictions["actual_class"].astype(str).eq("hard").to_numpy(int)
        probability = predictions["probability_hard"].to_numpy(float)
        false_positive, true_positive, _ = metric_functions["roc_curve"](
            truth, probability
        )
        auc = metric_functions["roc_auc_score"](truth, probability)
        color = MODEL_COLORS.get(model_key, "#555555")
        axes[0].plot(
            false_positive, true_positive, color=color, linewidth=2.2,
            label=f"{MODEL_DISPLAY[model_key]} ({auc:.3f})"
        )
        bin_ids = np.minimum((probability * 10).astype(int), 9)
        observed, predicted = [], []
        for bin_id in range(10):
            mask = bin_ids == bin_id
            if np.any(mask):
                observed.append(float(np.mean(truth[mask])))
                predicted.append(float(np.mean(probability[mask])))
        axes[1].plot(
            predicted, observed, marker="o", markersize=4, linewidth=1.8,
            color=color, label=MODEL_DISPLAY[model_key]
        )
        axes[2].hist(
            probability[truth == 0], bins=np.linspace(0, 1, 16),
            histtype="step", linewidth=1.7, color=color, alpha=0.7
        )
        axes[2].hist(
            probability[truth == 1], bins=np.linspace(0, 1, 16),
            histtype="stepfilled", color=color, alpha=0.12
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Discrimination")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    axes[1].set(xlabel="Mean predicted probability", ylabel="Observed hard fraction", title="Calibration")
    axes[2].set(xlabel="Predicted probability of hard", ylabel="OOF observations", title="Confidence distribution")
    axes[2].text(
        0.02, 0.98, "Outline: observed easy\nFill: observed hard",
        transform=axes[2].transAxes, ha="left", va="top", fontsize=8,
        color="#444444"
    )
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.grid(alpha=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Out-of-fold difficulty-classification diagnostics", fontsize=14)
    figure.tight_layout()
    save_figure(
        figure, task_dir / "figure_classification_diagnostics",
        figure_formats, figure_dpi, plt
    )


def plot_confusion_grid(
    predictions_by_model: dict[str, Any],
    task_dir: Path,
    plt: Any,
    np: Any,
    figure_formats: list[str],
    figure_dpi: int,
) -> None:
    model_items = list(predictions_by_model.items())
    columns = min(2, len(model_items))
    rows = int(np.ceil(len(model_items) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.4 * rows), squeeze=False)
    for axis, (model_key, predictions) in zip(axes.flat, model_items):
        truth = predictions["actual_class"].astype(str).to_numpy()
        predicted = predictions["predicted_class"].astype(str).to_numpy()
        matrix = np.asarray(
            [[np.sum((truth == actual) & (predicted == guessed)) for guessed in ("easy", "hard")]
             for actual in ("easy", "hard")], dtype=int
        )
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column, row, f"{matrix[row, column]}\n{normalized[row, column]:.1%}",
                    ha="center", va="center", fontsize=12,
                    color="white" if normalized[row, column] > 0.55 else "#1B1B1B"
                )
        axis.set_xticks([0, 1], ["Easy", "Hard"])
        axis.set_yticks([0, 1], ["Easy", "Hard"])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Observed class")
        axis.set_title(MODEL_DISPLAY[model_key])
    for axis in axes.flat[len(model_items):]:
        axis.set_visible(False)
    figure.colorbar(image, ax=list(axes.flat[:len(model_items)]), shrink=0.72, label="Row-normalized proportion")
    figure.suptitle("Out-of-fold confusion matrices", fontsize=14)
    figure.subplots_adjust(top=0.90, wspace=0.30, hspace=0.35)
    save_figure(
        figure, task_dir / "figure_confusion_matrices", figure_formats,
        figure_dpi, plt
    )


def plot_regression_diagnostics(
    predictions_by_model: dict[str, Any],
    task_dir: Path,
    plt: Any,
    np: Any,
    figure_formats: list[str],
    figure_dpi: int,
) -> None:
    model_items = list(predictions_by_model.items())
    figure, axes = plt.subplots(2, len(model_items), figsize=(4.2 * len(model_items), 8.2), squeeze=False)
    all_values = np.concatenate(
        [predictions[["actual_performance", "predicted_performance"]].to_numpy(float).ravel()
         for _, predictions in model_items]
    )
    lower, upper = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    for column, (model_key, predictions) in enumerate(model_items):
        actual = predictions["actual_performance"].to_numpy(float)
        predicted = predictions["predicted_performance"].to_numpy(float)
        residual = actual - predicted
        color = MODEL_COLORS.get(model_key, "#555555")
        axes[0, column].scatter(actual, predicted, s=22, alpha=0.55, color=color, edgecolors="none", rasterized=True)
        axes[0, column].plot([lower, upper], [lower, upper], "--", color="#444444", linewidth=1.2)
        axes[0, column].set_title(MODEL_DISPLAY[model_key])
        axes[0, column].set_xlabel("Observed performance")
        if column == 0:
            axes[0, column].set_ylabel("Predicted performance")
        axes[1, column].scatter(predicted, residual, s=22, alpha=0.55, color=color, edgecolors="none", rasterized=True)
        axes[1, column].axhline(0, color="#444444", linewidth=1.2, linestyle="--")
        axes[1, column].set_xlabel("Predicted performance")
        if column == 0:
            axes[1, column].set_ylabel("Residual (observed − predicted)")
        for axis in axes[:, column]:
            axis.grid(alpha=0.55)
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Out-of-fold performance-regression diagnostics", fontsize=14)
    figure.tight_layout()
    save_figure(
        figure, task_dir / "figure_regression_diagnostics", figure_formats,
        figure_dpi, plt
    )


def plot_tabfm_ensemble_diagnostics(
    task: str,
    predictions: Any,
    task_dir: Path,
    pd: Any,
    plt: Any,
    np: Any,
    figure_formats: list[str],
    figure_dpi: int,
) -> None:
    """Relate TabFM member disagreement to held-out prediction error."""
    working = predictions.copy()
    if task == "classification":
        uncertainty_column = (
            "ensemble_member_probability_hard_std_uncalibrated"
        )
        if uncertainty_column not in predictions:
            return
        uncertainty = predictions[uncertainty_column].to_numpy(float)
        correct_values = (
            predictions["correct"].astype(str).str.casefold().eq("true")
        )
        working["correct"] = correct_values
        error = (~correct_values).to_numpy(int)
        figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
        axes[0].scatter(
            predictions["probability_hard"], uncertainty,
            c=np.where(error == 0, "#009E73", "#D55E00"),
            s=30, alpha=0.65, edgecolors="none", rasterized=True
        )
        axes[0].set(
            xlabel="Final predicted probability of hard",
            ylabel="SD across 32 member probabilities",
            title="Run-level ensemble disagreement",
        )
        outcome_name = "accuracy"
    else:
        uncertainty_column = "ensemble_member_prediction_std"
        if uncertainty_column not in predictions:
            return
        uncertainty = predictions[uncertainty_column].to_numpy(float)
        error = predictions["absolute_error"].to_numpy(float)
        figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
        axes[0].scatter(
            uncertainty, error, color=MODEL_COLORS["tabfm_ensemble"],
            s=30, alpha=0.6, edgecolors="none", rasterized=True
        )
        axes[0].set(
            xlabel="SD across 32 member predictions",
            ylabel="Absolute OOF error",
            title="Disagreement versus error",
        )
        outcome_name = "mae"
    working["ensemble_uncertainty_quartile"] = pd.qcut(
        uncertainty,
        q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
        duplicates="drop",
    )
    if task == "classification":
        summary = (
            working.groupby("ensemble_uncertainty_quartile", observed=True)
            .agg(
                runs=("source_excel_row", "size"),
                accuracy=("correct", "mean"),
                mean_member_sd=(uncertainty_column, "mean"),
            )
            .reset_index()
        )
        axes[1].bar(
            summary["ensemble_uncertainty_quartile"].astype(str),
            summary["accuracy"], color="#0072B2"
        )
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("OOF accuracy")
    else:
        summary = (
            working.groupby("ensemble_uncertainty_quartile", observed=True)
            .agg(
                runs=("source_excel_row", "size"),
                mae=("absolute_error", "mean"),
                mean_member_sd=(uncertainty_column, "mean"),
            )
            .reset_index()
        )
        axes[1].bar(
            summary["ensemble_uncertainty_quartile"].astype(str),
            summary["mae"], color="#0072B2"
        )
        axes[1].set_ylabel("OOF MAE")
    summary.to_csv(
        task_dir / "tabfm_ensemble_disagreement_summary.csv", index=False
    )
    axes[1].set_xlabel("Ensemble-disagreement quartile")
    axes[1].set_title(f"Held-out {outcome_name} by disagreement")
    for axis in axes:
        axis.grid(axis="y", alpha=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "TabFM Ensemble: internal member disagreement as an uncertainty signal",
        fontsize=14,
    )
    figure.tight_layout()
    save_figure(
        figure,
        task_dir / "figure_tabfm_ensemble_disagreement",
        figure_formats,
        figure_dpi,
        plt,
    )


def plot_fold_stability(
    task: str,
    combined_folds: Any,
    task_dir: Path,
    plt: Any,
    figure_formats: list[str],
    figure_dpi: int,
) -> None:
    metric = "roc_auc_hard" if task == "classification" else "rmse"
    ylabel = "ROC-AUC (higher is better)" if task == "classification" else "RMSE (lower is better)"
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for model_key, group in combined_folds.groupby("model", sort=False):
        group = group.sort_values("fold")
        axis.plot(
            group["fold"], group[metric], marker="o", linewidth=2,
            color=MODEL_COLORS.get(str(model_key), "#555555"),
            label=MODEL_DISPLAY.get(str(model_key), str(model_key))
        )
    axis.set_xticks(sorted(combined_folds["fold"].unique()))
    axis.set_xlabel("Outer fold")
    axis.set_ylabel(ylabel)
    axis.set_title("Fold-to-fold stability on identical test partitions")
    axis.grid(axis="y")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=2)
    figure.tight_layout()
    save_figure(figure, task_dir / "figure_fold_stability", figure_formats, figure_dpi, plt)


def plot_feature_frequency(
    feature_frequency: Any,
    model_key: str,
    model_dir: Path,
    plt: Any,
    figure_formats: list[str],
    figure_dpi: int,
) -> None:
    top = feature_frequency.sort_values(
        ["selection_frequency", "mean_selection_rank"], ascending=[False, True]
    ).head(20).sort_values("selection_frequency")
    if top.empty:
        return
    modality_palette = {
        modality: color for modality, color in zip(
            MODALITY_ORDER,
            ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
             "#AA3377", "#BBBBBB", "#000000", "#EE8866", "#44AA99", "#999999"]
        )
    }
    figure, axis = plt.subplots(figsize=(9.5, 7.2))
    colors = [modality_palette.get(value, "#999999") for value in top["modality"]]
    axis.barh(top["feature"], top["selection_frequency"], color=colors, alpha=0.9)
    axis.set_xlim(0, 1.04)
    axis.set_xlabel("Fraction of outer folds selected")
    axis.set_title(f"Most consistently selected features — {MODEL_DISPLAY[model_key]}")
    axis.grid(axis="x")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", labelsize=8, length=0)
    present = list(dict.fromkeys(top["modality"].astype(str)))
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=modality_palette.get(value, "#999999"), label=value) for value in present]
    axis.legend(handles=handles, loc="lower right", fontsize=8)
    figure.tight_layout()
    save_figure(figure, model_dir / "figure_selected_feature_frequency", figure_formats, figure_dpi, plt)


def plot_modality_selection(
    modality_summary: Any,
    task_dir: Path,
    plt: Any,
    np: Any,
    figure_formats: list[str],
    figure_dpi: int,
) -> None:
    if modality_summary.empty:
        return
    pivot = modality_summary.pivot(index="modality", columns="model", values="mean_selection_fraction")
    ordered_modalities = [value for value in MODALITY_ORDER if value in pivot.index]
    ordered_models = [value for value in MODEL_CHOICES if value in pivot.columns]
    pivot = pivot.reindex(index=ordered_modalities, columns=ordered_models)
    figure, axis = plt.subplots(figsize=(7.5, max(4.8, 0.48 * len(pivot))))
    color_max = max(0.4, float(np.nanmax(pivot.to_numpy(float))))
    image = axis.imshow(
        pivot.to_numpy(float), cmap="YlGnBu", vmin=0,
        vmax=color_max, aspect="auto"
    )
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iloc[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.0%}", ha="center", va="center", color="white" if value > 0.55 else "#222222")
    axis.set_xticks(range(len(pivot.columns)), [MODEL_DISPLAY[value] for value in pivot.columns], rotation=20, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title("Fold-local feature-selection coverage by modality")
    figure.colorbar(image, ax=axis, shrink=0.78, label="Mean within-fold fraction selected")
    figure.tight_layout()
    save_figure(figure, task_dir / "figure_modality_selection", figure_formats, figure_dpi, plt)


def plot_dataset_overview(
    levels: Any,
    performance: Any,
    subjects: Any,
    output_dir: Path,
    pd: Any,
    plt: Any,
    np: Any,
    figure_formats: list[str],
    figure_dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    level_counts = pd.Series(levels).value_counts().sort_index()
    axes[0].bar(
        [str(value) for value in level_counts.index], level_counts.values,
        color=["#56B4E9", "#0072B2", "#E69F00", "#D55E00"]
    )
    axes[0].set(xlabel="Flight-difficulty level", ylabel="Session runs", title="Difficulty distribution")
    axes[0].axvline(1.5, color="#555555", linestyle="--", linewidth=1)
    axes[0].text(0.5, axes[0].get_ylim()[1] * 0.92, "Easy", ha="center", color="#0072B2")
    axes[0].text(2.5, axes[0].get_ylim()[1] * 0.92, "Hard", ha="center", color="#D55E00")
    axes[1].hist(performance, bins="auto", color="#0072B2", alpha=0.82, edgecolor="white")
    axes[1].axvline(float(np.mean(performance)), color="#D55E00", linestyle="--", linewidth=1.8, label="Mean")
    axes[1].set(xlabel="Landing performance", ylabel="Session runs", title="Performance distribution")
    axes[1].legend()
    pilot_counts = pd.Series(subjects).value_counts().sort_values()
    axes[2].barh(np.arange(len(pilot_counts)), pilot_counts.values, color="#009E73", alpha=0.85)
    axes[2].set(xlabel="Session runs per pilot", ylabel="Pilots (sorted)", title="Repeated-measures structure")
    axes[2].set_yticks([])
    for axis in axes:
        axis.grid(axis="y", alpha=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("VR landing-study dataset overview", fontsize=14)
    figure.tight_layout()
    save_figure(figure, output_dir / "figure_dataset_overview", figure_formats, figure_dpi, plt)


def plot_paired_advantages(
    task: str,
    paired: Any,
    task_dir: Path,
    plt: Any,
    np: Any,
    figure_formats: list[str],
    figure_dpi: int,
) -> None:
    if paired.empty:
        return
    metrics = (
        ["accuracy", "f1_macro", "roc_auc_hard"]
        if task == "classification"
        else ["mae", "rmse", "r2"]
    )
    labels = {
        "accuracy": "Accuracy", "f1_macro": "Macro F1",
        "roc_auc_hard": "ROC-AUC", "mae": "MAE", "rmse": "RMSE", "r2": "R²"
    }
    subset = paired.loc[paired["metric"].isin(metrics)].copy()
    comparators = [value for value in MODEL_CHOICES if value in set(subset["comparator_model"])]
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 4.5), squeeze=False)
    for index, metric in enumerate(metrics):
        axis = axes[0, index]
        metric_rows = subset.loc[subset["metric"].eq(metric)].set_index("comparator_model")
        y_positions = np.arange(len(comparators))[::-1]
        for position, comparator in enumerate(comparators):
            row = metric_rows.loc[comparator]
            point = float(row["point_advantage_tabfm"])
            axis.errorbar(
                point, y_positions[position],
                xerr=np.asarray(
                    [[max(0.0, point - float(row["ci_lower"]))],
                     [max(0.0, float(row["ci_upper"]) - point)]]
                ),
                fmt="o", capsize=3, linewidth=2,
                color=MODEL_COLORS.get(comparator, "#555555")
            )
        axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(labels[metric])
        axis.set_xlabel("Advantage; positive favors TabFM")
        axis.set_yticks(y_positions)
        axis.set_yticklabels([MODEL_DISPLAY[value] for value in comparators] if index == 0 else [])
        axis.grid(axis="x")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    figure.suptitle("Paired pilot-bootstrap differences versus TabFM Ensemble", fontsize=14)
    figure.tight_layout()
    save_figure(figure, task_dir / "figure_paired_advantages_vs_tabfm", figure_formats, figure_dpi, plt)


def transformed_selected_inputs(
    pipeline: Any,
    raw_frame: Any,
    selected_features: list[Any],
    pd: Any,
) -> Any:
    """Transform raw predictors to the exact selected feature space."""
    values = pipeline.named_steps["imputer"].transform(raw_frame)
    values = pipeline.named_steps["selector"].transform(values)
    scaler = pipeline.named_steps.get("scaler")
    if scaler is not None:
        values = scaler.transform(values)
    return pd.DataFrame(values, columns=[str(value) for value in selected_features])


def explain_fold_with_shap(
    task: str,
    model_key: str,
    fold_number: int,
    estimator: Any,
    train_input: Any,
    test_input: Any,
    test_indices: Any,
    source_excel_rows: Any,
    subjects: Any,
    output_path: Path,
    args: argparse.Namespace,
    shap: Any,
    pd: Any,
    np: Any,
) -> None:
    from direct_shap import explain_fold_with_shap as direct_explain
    return direct_explain(task, model_key, fold_number, estimator, train_input,
        test_input, test_indices, source_excel_rows, subjects, output_path, args, shap, pd, np)


def aggregate_shap_outputs(
    model_dir: Path,
    model_key: str,
    pd: Any,
    plt: Any,
    figure_formats: list[str],
    figure_dpi: int,
) -> Any:
    paths = sorted((model_dir / "folds").glob("fold_*_shap_values.csv"))
    if not paths:
        return None
    values = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    values.to_csv(
        model_dir / "shap_values_all_explained_rows.csv", index=False
    )
    explained_rows = int(values["source_excel_row"].nunique())
    predictions_path = model_dir / "oof_predictions.csv"
    total_oof_rows = (
        int(pd.read_csv(predictions_path)["source_excel_row"].nunique())
        if predictions_path.is_file()
        else None
    )
    complete_oof_coverage = (
        total_oof_rows is not None and explained_rows == total_oof_rows
    )
    if complete_oof_coverage:
        values.to_csv(model_dir / "oof_shap_values.csv", index=False)
    save_json(
        model_dir / "shap_coverage.json",
        {
            "explained_source_rows": explained_rows,
            "total_oof_source_rows": total_oof_rows,
            "complete_oof_coverage": complete_oof_coverage,
            "fold_files": [str(path) for path in paths],
            "interpretation": (
                "SHAP values cover every OOF row"
                if complete_oof_coverage
                else "SHAP values cover only the explicitly sampled/requested OOF rows"
            ),
        },
    )
    feature_importance = (
        values.assign(abs_shap=lambda value: value["shap_value"].abs())
        .groupby(["feature", "modality"], as_index=False)
        .agg(mean_abs_shap=("abs_shap", "mean"), mean_shap=("shap_value", "mean"), explained_rows=("source_excel_row", "nunique"))
        .sort_values("mean_abs_shap", ascending=False)
    )
    modality_importance = (
        feature_importance.groupby("modality", as_index=False)
        .agg(total_mean_abs_shap=("mean_abs_shap", "sum"), features_explained=("feature", "nunique"))
        .sort_values("total_mean_abs_shap", ascending=False)
    )
    feature_importance.to_csv(model_dir / "shap_feature_importance.csv", index=False)
    modality_importance.to_csv(model_dir / "shap_modality_importance.csv", index=False)
    top = feature_importance.head(20).sort_values("mean_abs_shap")
    figure, axis = plt.subplots(figsize=(9.5, 7.2))
    axis.barh(top["feature"], top["mean_abs_shap"], color=MODEL_COLORS.get(model_key, "#555555"))
    axis.set_xlabel("Mean |SHAP value| in model output units")
    axis.set_title(f"Fold-local permutation SHAP importance — {MODEL_DISPLAY[model_key]}")
    axis.grid(axis="x")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", labelsize=8, length=0)
    figure.tight_layout()
    save_figure(figure, model_dir / "figure_shap_feature_importance", figure_formats, figure_dpi, plt)
    return modality_importance.assign(model=model_key)


def selected_features_from_fold_audit(
    path: Path,
    allowed_features: set[str],
    pd: Any,
) -> list[str]:
    """Read the exact predictor subset used by one saved TabFM outer fold."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved TabFM feature audit not found: {path}. Run the benchmark "
            "before requesting fast-surrogate SHAP."
        )
    audit = pd.read_csv(path)
    required = {"feature", "selected"}
    missing = required.difference(audit.columns)
    if missing:
        raise RuntimeError(
            f"Feature audit lacks {sorted(missing)}: {path}"
        )
    selected_mask = (
        audit["selected"].astype(str).str.strip().str.casefold()
        .isin({"true", "1", "yes"})
    )
    selected = audit.loc[selected_mask, "feature"].astype(str).tolist()
    if not selected:
        raise RuntimeError(f"Feature audit selects no predictors: {path}")
    unknown = [feature for feature in selected if feature not in allowed_features]
    if unknown:
        raise RuntimeError(
            f"Saved selected features are absent from the current workbook: "
            f"{unknown[:20]}"
        )
    return selected


def surrogate_fidelity_metrics(
    observed: Any,
    predicted: Any,
    metric_functions: dict[str, Any],
    pd: Any,
    np: Any,
) -> dict[str, float]:
    """Metrics for how closely a cross-fitted surrogate reproduces TabFM."""
    observed_values = np.asarray(observed, dtype=float).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=float).reshape(-1)
    if len(observed_values) != len(predicted_values) or not len(observed_values):
        raise ValueError("Surrogate fidelity arrays must be non-empty and aligned.")
    observed_rank = pd.Series(observed_values).rank(method="average")
    predicted_rank = pd.Series(predicted_values).rank(method="average")
    pearson = (
        float(np.corrcoef(observed_values, predicted_values)[0, 1])
        if np.std(observed_values) > 0 and np.std(predicted_values) > 0
        else float("nan")
    )
    return {
        "n_rows": int(len(observed_values)),
        "r2": float(
            metric_functions["r2_score"](observed_values, predicted_values)
        ),
        "mae": float(
            metric_functions["mean_absolute_error"](
                observed_values, predicted_values
            )
        ),
        "rmse": float(
            np.sqrt(
                metric_functions["mean_squared_error"](
                    observed_values, predicted_values
                )
            )
        ),
        "pearson_r": pearson,
        "spearman_r": float(observed_rank.corr(predicted_rank)),
    }


def representative_rows_for_shap(
    task: str,
    fold_predictions: Any,
    output_column: str,
    requested_count: int,
    np: Any,
    sampling_strategy: str = "output_balanced",
) -> Any:
    """Deterministically span model output, optionally within four cohorts."""
    count = min(int(requested_count), len(fold_predictions))
    local_positions = np.arange(len(fold_predictions), dtype=int)

    def spaced(positions: Any, number: int) -> list[int]:
        if number <= 0 or not len(positions):
            return []
        ordered = np.asarray(positions, dtype=int)[
            np.argsort(
                fold_predictions.iloc[np.asarray(positions, dtype=int)][
                    output_column
                ].to_numpy(float),
                kind="mergesort",
            )
        ]
        if number >= len(ordered):
            return ordered.astype(int).tolist()
        indices = np.linspace(0, len(ordered) - 1, number)
        return ordered[np.rint(indices).astype(int)].astype(int).tolist()

    def spaced_distinct_pilots(positions: Any, number: int) -> list[int]:
        """Span output while preferring a different pilot for each row."""
        if number <= 0 or not len(positions):
            return []
        positions = np.asarray(positions, dtype=int)
        ordered = positions[
            np.argsort(
                fold_predictions.iloc[positions][output_column].to_numpy(float),
                kind="mergesort",
            )
        ]
        if number >= len(ordered):
            return ordered.astype(int).tolist()
        target_ranks = np.linspace(0, len(ordered) - 1, number)
        subject_column = (
            "subject" if "subject" in fold_predictions.columns else None
        )
        chosen_positions: list[int] = []
        used_subjects: set[str] = set()
        remaining = set(int(value) for value in ordered)
        rank_lookup = {
            int(position): rank for rank, position in enumerate(ordered)
        }
        for target_rank in target_ranks:
            candidates = sorted(
                remaining,
                key=lambda position: (
                    0
                    if subject_column is None
                    or str(fold_predictions.iloc[position][subject_column])
                    not in used_subjects
                    else 1,
                    abs(rank_lookup[position] - float(target_rank)),
                    rank_lookup[position],
                ),
            )
            selected_position = int(candidates[0])
            chosen_positions.append(selected_position)
            remaining.remove(selected_position)
            if subject_column is not None:
                used_subjects.add(
                    str(fold_predictions.iloc[selected_position][subject_column])
                )
        return chosen_positions

    chosen: list[int] = []
    if sampling_strategy == "experience_difficulty":
        required = {"flight_hours"}
        if not required.issubset(fold_predictions.columns):
            raise RuntimeError(
                "Experience/difficulty SHAP sampling requires flight_hours."
            )
        if "actual_class" in fold_predictions.columns:
            difficulty = (
                fold_predictions["actual_class"].astype(str).str.casefold()
            )
        elif "level" in fold_predictions.columns:
            levels = fold_predictions["level"].to_numpy(int)
            difficulty = np.where(levels <= 2, "easy", "hard")
        else:
            raise RuntimeError(
                "Experience/difficulty SHAP sampling requires actual_class "
                "or level."
            )
        hours = fold_predictions["flight_hours"].to_numpy(float)
        experience = np.where(
            hours <= 105.0,
            "novice",
            np.where(hours >= 1000.0, "experienced", "intermediate"),
        )
        allocation = count // 4
        for experience_name, class_name in (
            ("novice", "easy"),
            ("novice", "hard"),
            ("experienced", "easy"),
            ("experienced", "hard"),
        ):
            candidates = np.flatnonzero(
                (experience == experience_name)
                & (np.asarray(difficulty, dtype=str) == class_name)
            )
            if len(candidates) < allocation:
                raise RuntimeError(
                    f"Fold lacks {allocation} rows for {experience_name}/"
                    f"{class_name}: found {len(candidates)}."
                )
            chosen.extend(spaced_distinct_pilots(candidates, allocation))
    elif task == "classification" and "actual_class" in fold_predictions:
        classes = ["easy", "hard"]
        first_count = count // 2
        allocations = [first_count, count - first_count]
        for class_name, allocation in zip(classes, allocations):
            candidates = np.flatnonzero(
                fold_predictions["actual_class"].astype(str)
                .str.casefold().eq(class_name).to_numpy(bool)
            )
            chosen.extend(spaced(candidates, allocation))
    else:
        chosen.extend(spaced(local_positions, count))
    chosen = list(dict.fromkeys(int(value) for value in chosen))
    if len(chosen) < count and sampling_strategy != "experience_difficulty":
        remaining = [
            int(value) for value in spaced(local_positions, count)
            if int(value) not in chosen
        ]
        chosen.extend(remaining[: count - len(chosen)])
    return np.asarray(sorted(chosen[:count]), dtype=int)


def training_medoid(values: Any, np: Any) -> int:
    """Return an observed row nearest the robust multivariate centre."""
    array = np.asarray(values, dtype=float)
    centre = np.median(array, axis=0)
    q25 = np.percentile(array, 25, axis=0)
    q75 = np.percentile(array, 75, axis=0)
    scale = q75 - q25
    standard_deviation = np.std(array, axis=0)
    scale = np.where(scale > 1e-12, scale, standard_deviation)
    scale = np.where(scale > 1e-12, scale, 1.0)
    squared_distance = np.mean(((array - centre) / scale) ** 2, axis=1)
    return int(np.argmin(squared_distance))


def vectorized_medoid_permutation_shap(
    predict_output: Any,
    explained_values: Any,
    background_value: Any,
    seed: int,
    prediction_batch_size: int,
    feature_names: list[str],
    pd: Any,
    np: Any,
) -> tuple[Any, Any, Any, float]:
    """One forward/reverse permutation per row, vectorized across rows.

    This is the minimum-evaluation antithetic permutation estimator used by
    SHAP (2F+1 masked states per explained row), specialized to one observed
    medoid baseline. Stacking states across rows prevents TabFM from being
    called repeatedly with under-filled query batches.
    """
    explained = np.asarray(explained_values, dtype=float)
    background = np.asarray(background_value, dtype=float).reshape(-1)
    if explained.ndim != 2 or explained.shape[1] != len(background):
        raise ValueError("Explained rows and SHAP medoid are not aligned.")
    n_rows, n_features = explained.shape
    generator = np.random.default_rng(seed)
    states_per_row = 2 * n_features + 1
    all_states = np.empty(
        (n_rows * states_per_row, n_features), dtype=float
    )
    permutations: list[Any] = []
    for row_index in range(n_rows):
        permutation = generator.permutation(n_features)
        permutations.append(permutation)
        start = row_index * states_per_row
        current = background.copy()
        all_states[start] = current
        for step, feature_index in enumerate(permutation, start=1):
            current = current.copy()
            current[feature_index] = explained[row_index, feature_index]
            all_states[start + step] = current
        current = explained[row_index].copy()
        reverse_start = start + n_features + 1
        for step, feature_index in enumerate(permutation):
            current = current.copy()
            current[feature_index] = background[feature_index]
            all_states[reverse_start + step] = current

    predictions = np.empty(len(all_states), dtype=float)
    batch_size = int(prediction_batch_size)
    for batch_start in range(0, len(all_states), batch_size):
        batch_stop = min(batch_start + batch_size, len(all_states))
        batch_frame = pd.DataFrame(
            all_states[batch_start:batch_stop], columns=feature_names
        )
        predictions[batch_start:batch_stop] = np.asarray(
            predict_output(batch_frame), dtype=float
        ).reshape(-1)

    shap_values = np.zeros_like(explained, dtype=float)
    base_values = np.empty(n_rows, dtype=float)
    explained_outputs = np.empty(n_rows, dtype=float)
    maximum_residual = 0.0
    for row_index, permutation in enumerate(permutations):
        start = row_index * states_per_row
        forward = predictions[start : start + n_features + 1]
        reverse_after = predictions[
            start + n_features + 1 : start + states_per_row
        ].copy()
        # The final reverse state is identical to the forward baseline. Reuse
        # the same prediction so bfloat16 inference jitter at this duplicate
        # endpoint cannot create an artificial additivity residual.
        reverse_after[-1] = forward[0]
        forward_contribution = np.diff(forward)
        reverse_before = np.concatenate(
            [forward[-1:].copy(), reverse_after[:-1]]
        )
        reverse_contribution = reverse_before - reverse_after
        shap_values[row_index, permutation] = 0.5 * (
            forward_contribution + reverse_contribution
        )
        base_values[row_index] = float(forward[0])
        explained_outputs[row_index] = float(forward[-1])
        residual = abs(
            base_values[row_index] + shap_values[row_index].sum()
            - explained_outputs[row_index]
        )
        maximum_residual = max(maximum_residual, float(residual))
    return shap_values, base_values, explained_outputs, maximum_residual


def run_fast_tabfm_shap(
    task: str,
    frame: Any,
    task_features: list[Any],
    source_excel_rows: Any,
    output_dir: Path,
    tabfm_backbone: Any,
    tabfm_module: Any,
    args: argparse.Namespace,
    metric_functions: dict[str, Any],
    pd: Any,
    plt: Any,
    np: Any,
) -> dict[str, Any]:
    """Explain a fidelity-audited, low-cost TabFM ensemble approximation."""
    model_dir = output_dir / task / "models" / "tabfm_ensemble"
    predictions_path = model_dir / "oof_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Saved TabFM OOF predictions not found: {predictions_path}"
        )
    predictions = pd.read_csv(predictions_path)
    output_column = (
        "probability_hard" if task == "classification"
        else "predicted_performance"
    )
    required = {"source_excel_row", "fold", "model", output_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise RuntimeError(
            f"OOF predictions lack {sorted(missing)}: {predictions_path}"
        )
    predictions = predictions.loc[
        predictions["model"].astype(str).eq("tabfm_ensemble")
    ].copy()
    predictions["source_excel_row"] = predictions["source_excel_row"].astype(int)
    predictions["fold"] = predictions["fold"].astype(int)
    if predictions["source_excel_row"].duplicated().any():
        raise RuntimeError("Saved TabFM OOF predictions contain duplicate rows.")
    expected_rows = set(np.asarray(source_excel_rows, dtype=int).tolist())
    if set(predictions["source_excel_row"].tolist()) != expected_rows:
        raise RuntimeError(
            "fast_tabfm requires complete saved OOF prediction coverage."
        )
    predictions = predictions.sort_values("source_excel_row").reset_index(drop=True)
    frame_positions = predictions["source_excel_row"].to_numpy(int) - 2
    teacher_output = pd.to_numeric(
        predictions[output_column], errors="raise"
    ).to_numpy(float)
    fold_ids = sorted(predictions["fold"].unique().astype(int).tolist())
    allowed_features = {str(feature) for feature in task_features}
    union_features: list[str] = []
    fold_payloads: dict[int, dict[str, Any]] = {}
    approximation_output = np.full(len(predictions), np.nan, dtype=float)
    fold_fidelity: list[dict[str, Any]] = []

    for fold_number in fold_ids:
        test_rows = np.flatnonzero(
            predictions["fold"].to_numpy(int) == fold_number
        )
        train_rows = np.flatnonzero(
            predictions["fold"].to_numpy(int) != fold_number
        )
        selected = selected_features_from_fold_audit(
            model_dir / "folds" /
            f"fold_{fold_number:02d}_selected_features.csv",
            allowed_features,
            pd,
        )
        for feature in selected:
            if feature not in union_features:
                union_features.append(feature)
        raw_train = (
            frame.iloc[frame_positions[train_rows]][selected]
            .replace([np.inf, -np.inf], np.nan)
        )
        raw_test = (
            frame.iloc[frame_positions[test_rows]][selected]
            .replace([np.inf, -np.inf], np.nan)
        )
        medians = raw_train.median(axis=0, skipna=True)
        if medians.isna().any():
            missing_features = medians[medians.isna()].index.astype(str).tolist()
            raise RuntimeError(
                f"Fold {fold_number} has all-missing selected features: "
                f"{missing_features[:20]}"
            )
        train_input = raw_train.fillna(medians).reset_index(drop=True)
        test_input = raw_test.fillna(medians).reset_index(drop=True)
        y_train = (
            np.where(
                pd.to_numeric(
                    frame.iloc[frame_positions[train_rows]][args.level_column],
                    errors="raise",
                ).to_numpy(int) <= 2,
                "easy", "hard",
            )
            if task == "classification"
            else pd.to_numeric(
                frame.iloc[frame_positions[train_rows]][args.performance_column],
                errors="raise",
            ).to_numpy(float)
        )
        estimator_arguments = {
            "model": tabfm_backbone,
            "n_estimators": int(args.shap_fast_members),
            "max_num_features": 500,
            "max_num_rows": None,
            "batch_size": int(args.tabfm_batch_size),
            "random_state": int(args.seed),
            "enable_nnls": False,
            "verbose": False,
        }
        fit_started = time.monotonic()
        if task == "classification":
            estimator = tabfm_module.TabFMClassifier.ensemble(
                **estimator_arguments,
                binary_calibration_method="none",
                multiclass_calibration_method="none",
            )
        else:
            estimator = tabfm_module.TabFMRegressor.ensemble(
                **estimator_arguments
            )
        estimator.fit(train_input, y_train)

        if task == "classification":
            classes = np.asarray(estimator.classes_)
            hard_matches = np.flatnonzero(
                np.asarray(
                    [str(value).casefold() == "hard" or value == 1
                     for value in classes]
                )
            )
            hard_column = int(hard_matches[0]) if len(hard_matches) else 1

            def predict_output(values: Any, fitted: Any = estimator,
                               column: int = hard_column) -> Any:
                return np.asarray(
                    fitted.predict_proba(values), dtype=float
                )[:, column]
        else:
            def predict_output(values: Any, fitted: Any = estimator) -> Any:
                return np.asarray(fitted.predict(values), dtype=float).reshape(-1)

        fold_output = np.asarray(
            predict_output(test_input), dtype=float
        ).reshape(-1)
        approximation_output[test_rows] = fold_output
        metrics = surrogate_fidelity_metrics(
            teacher_output[test_rows], fold_output,
            metric_functions, pd, np,
        )
        metrics.update(
            {
                "fold": int(fold_number),
                "n_train_rows": int(len(train_rows)),
                "n_selected_features": int(len(selected)),
                "fit_and_full_fold_prediction_seconds": float(
                    time.monotonic() - fit_started
                ),
            }
        )
        fold_fidelity.append(metrics)
        fold_payloads[fold_number] = {
            "test_rows": test_rows,
            "train_input": train_input,
            "test_input": test_input,
            "selected": selected,
            "predict_output": predict_output,
            "estimator": estimator,
        }
        print(
            f"  fast TabFM audit {task} fold {fold_number:02d}: "
            f"members={args.shap_fast_members}, rows={len(test_rows)}, "
            f"R2={metrics['r2']:.3f}, MAE={metrics['mae']:.4g}",
            flush=True,
        )

    overall_fidelity = surrogate_fidelity_metrics(
        teacher_output, approximation_output, metric_functions, pd, np
    )
    manuscript_ready = bool(
        overall_fidelity["r2"] >= float(args.shap_fast_min_r2)
    )
    artifact_name = (
        "fast_tabfm_shap_experience_difficulty"
        if args.shap_fast_sampling == "experience_difficulty"
        else "fast_tabfm_shap"
    )
    artifact_dir = model_dir / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fidelity_frame = pd.DataFrame(fold_fidelity)
    overall_row = {
        **overall_fidelity,
        "fold": "overall_oof",
        "n_train_rows": int(len(predictions) - round(len(predictions) / len(fold_ids))),
        "n_selected_features": int(len(union_features)),
        "fit_and_full_fold_prediction_seconds": float(
            fidelity_frame["fit_and_full_fold_prediction_seconds"].sum()
        ),
    }
    fidelity_frame = pd.concat(
        [fidelity_frame, pd.DataFrame([overall_row])], ignore_index=True
    )
    atomic_write_csv(
        fidelity_frame, artifact_dir / "approximation_fidelity.csv"
    )
    atomic_write_csv(
        predictions[["source_excel_row", "fold"]].assign(
            full_ensemble_output=teacher_output,
            fast_tabfm_output=approximation_output,
            residual_full_minus_fast=teacher_output - approximation_output,
        ),
        artifact_dir / "approximation_oof_predictions.csv",
    )

    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    fold_palette = ["#0072B2", "#E69F00", "#D55E00", "#7A3E9D", "#7F8C3A"]
    for color_index, fold_number in enumerate(fold_ids):
        mask = predictions["fold"].to_numpy(int) == fold_number
        axis.scatter(
            teacher_output[mask], approximation_output[mask],
            s=28, alpha=0.72, linewidth=0.35, edgecolor="white",
            color=fold_palette[color_index % len(fold_palette)],
            label=f"Fold {fold_number}",
        )
    lower = float(min(np.min(teacher_output), np.min(approximation_output)))
    upper = float(max(np.max(teacher_output), np.max(approximation_output)))
    padding = max((upper - lower) * 0.04, 1e-6)
    axis.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        color="#333333", linestyle="--", linewidth=1.2,
        label="Ideal agreement",
    )
    axis.set_xlim(lower - padding, upper + padding)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_aspect("equal", adjustable="box")
    unit_label = (
        "hard-class probability" if task == "classification"
        else "predicted performance"
    )
    axis.set_xlabel(f"Saved 32-member TabFM OOF {unit_label}")
    axis.set_ylabel(
        f"{args.shap_fast_members}-member fast TabFM {unit_label}"
    )
    axis.set_title("Fast TabFM approximation fidelity", loc="left")
    axis.text(
        0.02, 0.98,
        f"R² = {overall_fidelity['r2']:.3f}\n"
        f"MAE = {overall_fidelity['mae']:.4g}\n"
        f"n = {len(predictions)}",
        transform=axis.transAxes, ha="left", va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9,
              "edgecolor": "#D8D8D8"},
    )
    axis.legend(loc="lower right", fontsize=8, ncol=2)
    axis.grid(True, alpha=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save_figure(
        figure, artifact_dir / "figure_approximation_fidelity",
        args.figure_formats, args.figure_dpi, plt,
    )

    if not manuscript_ready and not args.allow_low_fidelity_shap:
        raise RuntimeError(
            f"Fast TabFM approximation fidelity is below threshold: "
            f"R2={overall_fidelity['r2']:.3f} < {args.shap_fast_min_r2:.3f}. "
            f"Fidelity artifacts were saved to {artifact_dir}; SHAP was not "
            "computed. Increase --shap-fast-members or use direct permutation "
            "SHAP."
        )

    union_index = {
        feature: index for index, feature in enumerate(union_features)
    }
    explained_records: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    maximum_additivity_residual = 0.0
    for fold_number in fold_ids:
        payload = fold_payloads[fold_number]
        test_rows = payload["test_rows"]
        fold_predictions = predictions.iloc[test_rows].reset_index(drop=True)
        local_positions = representative_rows_for_shap(
            task, fold_predictions, output_column,
            args.shap_fast_rows_per_fold, np,
            sampling_strategy=args.shap_fast_sampling,
        )
        selected = payload["selected"]
        train_values = payload["train_input"].to_numpy(float)
        medoid_position = training_medoid(train_values, np)
        background = train_values[medoid_position]
        explained_values = payload["test_input"].iloc[
            local_positions
        ].to_numpy(float)
        fold_shap, base, explained_output, residual = (
            vectorized_medoid_permutation_shap(
                predict_output=payload["predict_output"],
                explained_values=explained_values,
                background_value=background,
                seed=int(args.seed + 20000 + fold_number),
                prediction_batch_size=int(args.shap_batch_size),
                feature_names=selected,
                pd=pd,
                np=np,
            )
        )
        maximum_additivity_residual = max(
            maximum_additivity_residual, float(residual)
        )
        global_rows = test_rows[local_positions]
        selected_indices = np.asarray(
            [union_index[feature] for feature in selected], dtype=int
        )
        for local_index, global_row in enumerate(global_rows):
            explained_records.append(
                {
                    "global_row": int(global_row),
                    "fold": int(fold_number),
                    "source_excel_row": int(
                        predictions.iloc[global_row]["source_excel_row"]
                    ),
                    "selected_indices": selected_indices,
                    "selected_features": selected,
                    "feature_values": explained_values[local_index],
                    "shap_values": fold_shap[local_index],
                    "base_value": float(base[local_index]),
                    "fast_tabfm_output": float(explained_output[local_index]),
                    "full_ensemble_output": float(teacher_output[global_row]),
                }
            )
        fold_results.append(
            {
                "fold": int(fold_number),
                "explained_rows": int(len(global_rows)),
                "selected_features": int(len(selected)),
                "background": "one observed robust training medoid",
                "masked_states": int(
                    len(global_rows) * (2 * len(selected) + 1)
                ),
            }
        )
        print(
            f"  fast TabFM SHAP {task} fold {fold_number:02d}: "
            f"explained={len(global_rows)}, features={len(selected)}, "
            f"masked_states={len(global_rows) * (2 * len(selected) + 1)}",
            flush=True,
        )

    n_explained = len(explained_records)
    sampling_rows = predictions.iloc[
        [record["global_row"] for record in explained_records]
    ].copy()
    sampling_rows["experience_group"] = np.where(
        sampling_rows["flight_hours"].to_numpy(float) <= 105.0,
        "novice",
        np.where(
            sampling_rows["flight_hours"].to_numpy(float) >= 1000.0,
            "experienced",
            "intermediate",
        ),
    )
    if "actual_class" in sampling_rows.columns:
        sampling_rows["difficulty_group"] = (
            sampling_rows["actual_class"].astype(str).str.casefold()
        )
    else:
        sampling_rows["difficulty_group"] = np.where(
            sampling_rows["level"].to_numpy(int) <= 2, "easy", "hard"
        )
    atomic_write_csv(
        sampling_rows,
        artifact_dir / "representative_sampling_rows.csv",
    )
    sampling_coverage = []
    for (fold_number, experience_name, difficulty_name), subset in (
        sampling_rows.groupby(
            ["fold", "experience_group", "difficulty_group"], sort=True
        )
    ):
        subject_column = "subject" if "subject" in subset.columns else None
        sampling_coverage.append(
            {
                "fold": int(fold_number),
                "experience_group": str(experience_name),
                "difficulty_group": str(difficulty_name),
                "rows": int(len(subset)),
                "unique_pilots": (
                    int(subset[subject_column].nunique())
                    if subject_column is not None else None
                ),
            }
        )
    shap_matrix = np.zeros(
        (n_explained, len(union_features)), dtype=float
    )
    aligned_raw = (
        frame.iloc[[
            frame_positions[record["global_row"]]
            for record in explained_records
        ]][union_features]
        .replace([np.inf, -np.inf], np.nan)
        .reset_index(drop=True)
    )
    global_medians = aligned_raw.median(axis=0, skipna=True).fillna(0.0)
    feature_value_matrix = aligned_raw.fillna(global_medians).to_numpy(float)
    base_values = np.empty(n_explained, dtype=float)
    fast_outputs = np.empty(n_explained, dtype=float)
    full_outputs = np.empty(n_explained, dtype=float)
    long_frames: list[Any] = []
    for row_index, record in enumerate(explained_records):
        indices = record["selected_indices"]
        shap_matrix[row_index, indices] = record["shap_values"]
        feature_value_matrix[row_index, indices] = record["feature_values"]
        base_values[row_index] = record["base_value"]
        fast_outputs[row_index] = record["fast_tabfm_output"]
        full_outputs[row_index] = record["full_ensemble_output"]
        long_frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "model": "tabfm_ensemble",
                    "fold": int(record["fold"]),
                    "source_excel_row": int(record["source_excel_row"]),
                    "feature": record["selected_features"],
                    "modality": [
                        assign_modality(feature)
                        for feature in record["selected_features"]
                    ],
                    "feature_value": record["feature_values"],
                    "shap_value": record["shap_values"],
                    "base_value": float(record["base_value"]),
                    "fast_tabfm_output": float(record["fast_tabfm_output"]),
                    "full_ensemble_output": float(
                        record["full_ensemble_output"]
                    ),
                    "explanation_method": (
                        "fidelity-audited fast TabFM antithetic permutation SHAP"
                    ),
                }
            )
        )
    atomic_write_csv(
        pd.concat(long_frames, ignore_index=True),
        artifact_dir / "shap_values.csv",
    )
    importance = pd.DataFrame(
        {
            "feature": union_features,
            "modality": [assign_modality(feature) for feature in union_features],
            "mean_abs_shap": np.mean(np.abs(shap_matrix), axis=0),
            "mean_shap": np.mean(shap_matrix, axis=0),
            "nonzero_attribution_rows": np.sum(
                ~np.isclose(shap_matrix, 0.0), axis=0
            ).astype(int),
            "explained_rows": int(n_explained),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    atomic_write_csv(
        importance, artifact_dir / "shap_feature_importance.csv"
    )
    np.savez_compressed(
        artifact_dir / "shap_matrix.npz",
        shap_values=shap_matrix,
        base_values=base_values,
        feature_values=feature_value_matrix,
        feature_names=np.asarray(union_features, dtype=object),
        source_excel_rows=np.asarray(
            [record["source_excel_row"] for record in explained_records],
            dtype=int,
        ),
        fold=np.asarray(
            [record["fold"] for record in explained_records], dtype=int
        ),
        fast_tabfm_output=fast_outputs,
        full_ensemble_output=full_outputs,
    )

    status_text = (
        "fidelity-audited fast TabFM permutation SHAP"
        if manuscript_ready else
        "EXPLORATORY: low-fidelity fast TabFM permutation SHAP"
    )
    import shap as imported_shap
    explanation = imported_shap.Explanation(
        values=shap_matrix,
        base_values=base_values,
        data=feature_value_matrix,
        feature_names=union_features,
    )
    imported_shap.plots.beeswarm(
        explanation,
        max_display=min(int(args.shap_top_features), len(union_features)),
        show=False,
        plot_size=(13.5, 9.2),
    )
    beeswarm_figure = plt.gcf()
    beeswarm_axis = beeswarm_figure.axes[0]
    beeswarm_axis.set_facecolor("white")
    beeswarm_axis.set_xlabel(
        "Fast TabFM SHAP value (impact on hard-class probability)"
        if task == "classification"
        else "Fast TabFM SHAP value (impact on predicted performance)"
    )
    beeswarm_axis.set_title(
        f"TabFM Ensemble — {TASK_DISPLAY[task]} SHAP summary\n"
        f"{status_text}; n={n_explained}; ensemble-fidelity R²="
        f"{overall_fidelity['r2']:.3f}",
        loc="left", pad=18,
    )
    beeswarm_axis.axvline(0.0, color="#333333", linewidth=0.9, zorder=0)
    beeswarm_axis.grid(axis="x", alpha=0.55)
    beeswarm_figure.subplots_adjust(
        left=0.42, right=0.94, top=0.86, bottom=0.10
    )
    save_figure(
        beeswarm_figure, artifact_dir / "figure_shap_beeswarm",
        args.figure_formats, args.figure_dpi, plt,
    )

    top = importance.head(int(args.shap_top_features)).sort_values(
        "mean_abs_shap", ascending=True
    )
    figure, axis = plt.subplots(
        figsize=(12.2, max(6.4, 0.38 * len(top) + 2.0))
    )
    axis.barh(
        top["feature"], top["mean_abs_shap"],
        color="#0072B2", edgecolor="#12405F", linewidth=0.55,
    )
    axis.set_xlabel(
        "Mean |fast TabFM SHAP value| (probability units)"
        if task == "classification"
        else "Mean |fast TabFM SHAP value| (performance units)"
    )
    axis.set_title(
        f"TabFM Ensemble — {TASK_DISPLAY[task]} global feature importance\n"
        f"{status_text}; n={n_explained}; ensemble-fidelity R²="
        f"{overall_fidelity['r2']:.3f}",
        loc="left", pad=14,
    )
    axis.grid(axis="x", alpha=0.55)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", labelsize=8.5, length=0)
    figure.tight_layout()
    save_figure(
        figure, artifact_dir / "figure_shap_mean_abs_bar",
        args.figure_formats, args.figure_dpi, plt,
    )

    methodology = {
        "task": task,
        "target_model": "32-member TabFM Ensemble saved OOF output",
        "explanation_model": (
            f"{args.shap_fast_members}-member TabFM approximation without "
            "NNLS weighting or output calibration"
        ),
        "not_direct_full_ensemble_shap": True,
        "method": (
            "vectorized single-medoid antithetic permutation SHAP "
            "(2F+1 masked states per explained row)"
        ),
        "representative_rows_per_fold": int(args.shap_fast_rows_per_fold),
        "representative_sampling_strategy": args.shap_fast_sampling,
        "experience_group_definition": {
            "novice": "flight_hours <= 105",
            "experienced": "flight_hours >= 1000",
            "intermediate": "105 < flight_hours < 1000",
        },
        "sampling_coverage": sampling_coverage,
        "explained_rows": int(n_explained),
        "full_oof_rows_used_for_fidelity": int(len(predictions)),
        "approximation_fidelity": overall_fidelity,
        "required_minimum_r2": float(args.shap_fast_min_r2),
        "manuscript_ready": manuscript_ready,
        "low_fidelity_override_used": bool(
            args.allow_low_fidelity_shap and not manuscript_ready
        ),
        "selected_features_by_fold": {
            str(record["fold"]): int(record["selected_features"])
            for record in fold_results
        },
        "background": "one observed robust medoid from each outer-training fold",
        "maximum_shap_additivity_residual": float(
            maximum_additivity_residual
        ),
        "interpretation_guardrail": (
            "The plot is a high-fidelity, model-native approximation of the "
            "32-member ensemble, not direct SHAP of the weighted/calibrated "
            "full ensemble. State the member count and OOF fidelity in the "
            "manuscript. Direct fold-local permutation SHAP remains the "
            "authoritative method for individual full-ensemble predictions."
        ),
        "fold_computation": fold_results,
    }
    save_json(artifact_dir / "methodology.json", methodology)
    print(
        f"  fast TabFM SHAP {task}: approximation R2="
        f"{overall_fidelity['r2']:.3f}; explained_rows={n_explained}; "
        f"beeswarm={artifact_dir}",
        flush=True,
    )
    return methodology


def run_fast_surrogate_shap(
    task: str,
    frame: Any,
    task_features: list[Any],
    source_excel_rows: Any,
    output_dir: Path,
    args: argparse.Namespace,
    shap: Any,
    ExtraTreesRegressor: Any,
    SimpleImputer: Any,
    metric_functions: dict[str, Any],
    pd: Any,
    plt: Any,
    np: Any,
) -> dict[str, Any]:
    """Create fast global SHAP via fidelity-audited cross-fitted surrogates.

    Direct permutation SHAP remains the authoritative local method. This fast
    route is intended for the global beeswarm commonly used in ML/XAI papers:
    it learns a separate Extra Trees surrogate for each original outer fold,
    excludes that fold's rows during surrogate fitting, uses exactly the
    predictors selected by that fold's TabFM model, and explains all held-out
    rows with TreeSHAP. The resulting figure is exported only after fidelity
    is quantified and (unless explicitly overridden) passes the requested R2.
    """
    model_dir = output_dir / task / "models" / "tabfm_ensemble"
    predictions_path = model_dir / "oof_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Saved TabFM OOF predictions not found: {predictions_path}. "
            "Run the benchmark once before fast-surrogate SHAP."
        )
    predictions = pd.read_csv(predictions_path)
    output_column = (
        "probability_hard" if task == "classification"
        else "predicted_performance"
    )
    required_prediction_columns = {
        "source_excel_row", "fold", "model", output_column
    }
    missing_prediction_columns = required_prediction_columns.difference(
        predictions.columns
    )
    if missing_prediction_columns:
        raise RuntimeError(
            f"OOF predictions lack {sorted(missing_prediction_columns)}: "
            f"{predictions_path}"
        )
    predictions = predictions.loc[
        predictions["model"].astype(str).eq("tabfm_ensemble")
    ].copy()
    if predictions.empty:
        raise RuntimeError(f"No TabFM rows found in {predictions_path}")
    predictions["source_excel_row"] = predictions["source_excel_row"].astype(int)
    predictions["fold"] = predictions["fold"].astype(int)
    if predictions["source_excel_row"].duplicated().any():
        duplicates = predictions.loc[
            predictions["source_excel_row"].duplicated(keep=False),
            "source_excel_row",
        ].astype(int).tolist()
        raise RuntimeError(
            f"OOF predictions contain duplicate source rows: {duplicates[:20]}"
        )
    expected_rows = set(np.asarray(source_excel_rows, dtype=int).tolist())
    saved_rows = set(predictions["source_excel_row"].astype(int).tolist())
    if saved_rows != expected_rows:
        raise RuntimeError(
            "Fast-surrogate SHAP requires complete OOF prediction coverage; "
            f"missing={sorted(expected_rows - saved_rows)[:20]}, "
            f"unexpected={sorted(saved_rows - expected_rows)[:20]}."
        )
    predictions = predictions.sort_values("source_excel_row").reset_index(drop=True)
    frame_positions = predictions["source_excel_row"].to_numpy(int) - 2
    if np.any(frame_positions < 0) or np.any(frame_positions >= len(frame)):
        raise RuntimeError("OOF source rows do not align with the current workbook.")
    target_output = pd.to_numeric(
        predictions[output_column], errors="raise"
    ).to_numpy(float)
    if not np.isfinite(target_output).all():
        raise RuntimeError(f"{output_column} contains non-finite OOF values.")

    fold_ids = sorted(predictions["fold"].unique().astype(int).tolist())
    allowed_features = {str(feature) for feature in task_features}
    selected_by_fold: dict[int, list[str]] = {}
    union_features: list[str] = []
    for fold_number in fold_ids:
        selected_path = (
            model_dir / "folds" /
            f"fold_{fold_number:02d}_selected_features.csv"
        )
        selected = selected_features_from_fold_audit(
            selected_path, allowed_features, pd
        )
        selected_by_fold[fold_number] = selected
        for feature in selected:
            if feature not in union_features:
                union_features.append(feature)
    if not union_features:
        raise RuntimeError("No selected TabFM predictors were available for SHAP.")

    aligned_raw = (
        frame.iloc[frame_positions][union_features]
        .replace([np.inf, -np.inf], np.nan)
        .reset_index(drop=True)
    )
    global_medians = aligned_raw.median(axis=0, skipna=True)
    all_missing = global_medians[global_medians.isna()].index.astype(str).tolist()
    if all_missing:
        raise RuntimeError(
            f"Selected features are entirely missing: {all_missing[:20]}"
        )
    feature_values = aligned_raw.fillna(global_medians).to_numpy(float)
    n_rows = len(predictions)
    n_union_features = len(union_features)
    shap_matrix = np.zeros((n_rows, n_union_features), dtype=float)
    base_values = np.full(n_rows, np.nan, dtype=float)
    surrogate_output = np.full(n_rows, np.nan, dtype=float)
    union_index = {
        feature: index for index, feature in enumerate(union_features)
    }
    long_frames: list[Any] = []
    fold_fidelity: list[dict[str, Any]] = []
    max_additivity_residual = 0.0

    for fold_number in fold_ids:
        test_mask = predictions["fold"].to_numpy(int) == fold_number
        train_mask = ~test_mask
        test_rows = np.flatnonzero(test_mask)
        train_rows = np.flatnonzero(train_mask)
        selected = selected_by_fold[fold_number]
        raw_train = (
            frame.iloc[frame_positions[train_rows]][selected]
            .replace([np.inf, -np.inf], np.nan)
        )
        raw_test = (
            frame.iloc[frame_positions[test_rows]][selected]
            .replace([np.inf, -np.inf], np.nan)
        )
        imputer = SimpleImputer(strategy="median")
        train_values = imputer.fit_transform(raw_train)
        test_values = imputer.transform(raw_test)
        surrogate = ExtraTreesRegressor(
            n_estimators=int(args.shap_surrogate_trees),
            min_samples_leaf=int(args.shap_surrogate_min_samples_leaf),
            max_features=1.0,
            bootstrap=False,
            random_state=int(args.seed + 10000 + fold_number),
            n_jobs=int(args.n_jobs),
        )
        fit_started = time.monotonic()
        surrogate.fit(train_values, target_output[train_rows])
        fold_surrogate_output = np.asarray(
            surrogate.predict(test_values), dtype=float
        ).reshape(-1)
        surrogate_output[test_rows] = fold_surrogate_output
        fold_metrics = surrogate_fidelity_metrics(
            target_output[test_rows], fold_surrogate_output,
            metric_functions, pd, np,
        )
        fold_metrics.update(
            {
                "fold": int(fold_number),
                "n_train_rows": int(len(train_rows)),
                "n_selected_features": int(len(selected)),
                "fit_and_predict_seconds": float(
                    time.monotonic() - fit_started
                ),
            }
        )
        fold_fidelity.append(fold_metrics)

        tree_explainer = shap.TreeExplainer(surrogate)
        fold_shap_values = np.asarray(
            tree_explainer.shap_values(test_values), dtype=float
        )
        if fold_shap_values.ndim == 3 and fold_shap_values.shape[-1] == 1:
            fold_shap_values = fold_shap_values[..., 0]
        expected_shape = (len(test_rows), len(selected))
        if fold_shap_values.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected surrogate SHAP shape {fold_shap_values.shape}; "
                f"expected {expected_shape}."
            )
        expected_value = float(
            np.asarray(tree_explainer.expected_value, dtype=float)
            .reshape(-1)[0]
        )
        base_values[test_rows] = expected_value
        additivity_residual = np.abs(
            expected_value + fold_shap_values.sum(axis=1)
            - fold_surrogate_output
        )
        max_additivity_residual = max(
            max_additivity_residual,
            float(np.max(additivity_residual)),
        )
        selected_indices = np.asarray(
            [union_index[feature] for feature in selected], dtype=int
        )
        shap_matrix[np.ix_(test_rows, selected_indices)] = fold_shap_values
        feature_values[np.ix_(test_rows, selected_indices)] = test_values

        repeated_rows = np.repeat(test_rows, len(selected))
        tiled_feature_positions = np.tile(
            np.arange(len(selected), dtype=int), len(test_rows)
        )
        long_frames.append(
            pd.DataFrame(
                {
                    "task": task,
                    "model": "tabfm_ensemble",
                    "fold": int(fold_number),
                    "source_excel_row": predictions.iloc[repeated_rows][
                        "source_excel_row"
                    ].to_numpy(int),
                    "subject": frame.iloc[
                        frame_positions[repeated_rows]
                    ][args.subject_column].to_numpy(),
                    "feature": np.asarray(selected, dtype=object)[
                        tiled_feature_positions
                    ],
                    "modality": [
                        assign_modality(selected[position])
                        for position in tiled_feature_positions
                    ],
                    "feature_value": test_values.reshape(-1),
                    "shap_value": fold_shap_values.reshape(-1),
                    "base_value": expected_value,
                    output_column: np.repeat(
                        target_output[test_rows], len(selected)
                    ),
                    "surrogate_output": np.repeat(
                        fold_surrogate_output, len(selected)
                    ),
                    "explanation_method": (
                        "cross-fitted Extra Trees surrogate TreeSHAP"
                    ),
                    "selected_in_fold": True,
                }
            )
        )
        print(
            f"  fast SHAP {task} fold {fold_number:02d}: "
            f"rows={len(test_rows)}, features={len(selected)}, "
            f"R2={fold_metrics['r2']:.3f}, "
            f"MAE={fold_metrics['mae']:.4g}",
            flush=True,
        )

    if not np.isfinite(surrogate_output).all() or not np.isfinite(base_values).all():
        raise RuntimeError("Fast-surrogate SHAP did not cover every OOF row.")
    overall_fidelity = surrogate_fidelity_metrics(
        target_output, surrogate_output, metric_functions, pd, np
    )
    manuscript_ready = bool(
        overall_fidelity["r2"] >= float(args.shap_surrogate_min_r2)
    )
    export_allowed = manuscript_ready or bool(args.allow_low_fidelity_shap)
    artifact_dir = model_dir / "fast_shap_surrogate"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fidelity_frame = pd.DataFrame(fold_fidelity)
    overall_row = {
        **overall_fidelity,
        "fold": "overall_cross_fitted",
        "n_train_rows": int(n_rows - round(n_rows / len(fold_ids))),
        "n_selected_features": int(n_union_features),
        "fit_and_predict_seconds": float(
            fidelity_frame["fit_and_predict_seconds"].sum()
        ),
    }
    fidelity_frame = pd.concat(
        [fidelity_frame, pd.DataFrame([overall_row])], ignore_index=True
    )
    atomic_write_csv(
        fidelity_frame,
        artifact_dir / "cross_fitted_surrogate_fidelity.csv",
    )
    shap_long = pd.concat(long_frames, ignore_index=True)
    atomic_write_csv(shap_long, artifact_dir / "shap_values.csv")

    selection_count = {
        feature: int(
            sum(feature in selected for selected in selected_by_fold.values())
        )
        for feature in union_features
    }
    importance = pd.DataFrame(
        {
            "feature": union_features,
            "modality": [assign_modality(feature) for feature in union_features],
            "mean_abs_shap": np.mean(np.abs(shap_matrix), axis=0),
            "mean_shap": np.mean(shap_matrix, axis=0),
            "selection_fold_count": [
                selection_count[feature] for feature in union_features
            ],
            "nonzero_attribution_rows": np.sum(
                ~np.isclose(shap_matrix, 0.0), axis=0
            ).astype(int),
            "total_oof_rows": int(n_rows),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    atomic_write_csv(
        importance, artifact_dir / "shap_feature_importance.csv"
    )
    np.savez_compressed(
        artifact_dir / "shap_matrix.npz",
        shap_values=shap_matrix,
        base_values=base_values,
        feature_values=feature_values,
        feature_names=np.asarray(union_features, dtype=object),
        source_excel_rows=predictions["source_excel_row"].to_numpy(int),
        fold=predictions["fold"].to_numpy(int),
        tabfm_oof_output=target_output,
        surrogate_output=surrogate_output,
    )

    methodology = {
        "task": task,
        "explained_model": "TabFM Ensemble OOF scalar output",
        "output_column": output_column,
        "method": "cross-fitted Extra Trees surrogate TreeSHAP",
        "not_direct_tabfm_shap": True,
        "training_rule": (
            "For each original outer fold, the surrogate was trained on rows "
            "from the other folds and restricted to that fold's saved TabFM "
            "selected predictors. TreeSHAP was then computed for the held-out "
            "fold and all fold explanations were aligned; predictors absent "
            "from a fold were assigned zero contribution for that fold."
        ),
        "surrogate_trees_per_fold": int(args.shap_surrogate_trees),
        "surrogate_min_samples_leaf": int(
            args.shap_surrogate_min_samples_leaf
        ),
        "rows_explained": int(n_rows),
        "union_features": int(n_union_features),
        "selected_features_by_fold": {
            str(fold): int(len(features))
            for fold, features in selected_by_fold.items()
        },
        "cross_fitted_fidelity": overall_fidelity,
        "required_minimum_r2": float(args.shap_surrogate_min_r2),
        "manuscript_ready": manuscript_ready,
        "low_fidelity_override_used": bool(
            args.allow_low_fidelity_shap and not manuscript_ready
        ),
        "maximum_tree_shap_additivity_residual": float(
            max_additivity_residual
        ),
        "interpretation_guardrail": (
            "These are surrogate TreeSHAP values approximating the saved "
            "TabFM OOF output surface. They must be labeled surrogate SHAP, "
            "and direct fold-local permutation SHAP remains the authoritative "
            "method for individual TabFM predictions."
        ),
    }
    save_json(artifact_dir / "methodology.json", methodology)

    fold_palette = ["#0072B2", "#E69F00", "#D55E00", "#7A3E9D", "#7F8C3A"]
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    for color_index, fold_number in enumerate(fold_ids):
        mask = predictions["fold"].to_numpy(int) == fold_number
        axis.scatter(
            target_output[mask], surrogate_output[mask],
            s=28, alpha=0.72, linewidth=0.35, edgecolor="white",
            color=fold_palette[color_index % len(fold_palette)],
            label=f"Fold {fold_number}",
        )
    lower = float(min(np.min(target_output), np.min(surrogate_output)))
    upper = float(max(np.max(target_output), np.max(surrogate_output)))
    padding = max((upper - lower) * 0.04, 1e-6)
    axis.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        color="#333333", linestyle="--", linewidth=1.2,
        label="Ideal agreement",
    )
    axis.set_xlim(lower - padding, upper + padding)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_aspect("equal", adjustable="box")
    unit_label = (
        "hard-class probability" if task == "classification"
        else "predicted performance"
    )
    axis.set_xlabel(f"Saved TabFM OOF {unit_label}")
    axis.set_ylabel(f"Cross-fitted surrogate {unit_label}")
    axis.set_title("TabFM-surrogate fidelity", loc="left")
    axis.text(
        0.02, 0.98,
        f"R² = {overall_fidelity['r2']:.3f}\n"
        f"MAE = {overall_fidelity['mae']:.4g}\n"
        f"n = {n_rows}",
        transform=axis.transAxes, ha="left", va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9,
              "edgecolor": "#D8D8D8"},
    )
    axis.legend(loc="lower right", fontsize=8, ncol=2)
    axis.grid(True, alpha=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save_figure(
        figure, artifact_dir / "figure_surrogate_fidelity",
        args.figure_formats, args.figure_dpi, plt,
    )

    if not export_allowed:
        raise RuntimeError(
            f"Fast-surrogate SHAP fidelity is below the manuscript threshold: "
            f"cross-fitted R2={overall_fidelity['r2']:.3f} < "
            f"{args.shap_surrogate_min_r2:.3f}. Fidelity artifacts were saved "
            f"to {artifact_dir}. Improve the surrogate or use direct "
            "permutation SHAP; use --allow-low-fidelity-shap only for a "
            "clearly exploratory figure."
        )

    explanation = shap.Explanation(
        values=shap_matrix,
        base_values=base_values,
        data=feature_values,
        feature_names=union_features,
    )
    shap.plots.beeswarm(
        explanation,
        max_display=min(int(args.shap_top_features), n_union_features),
        show=False,
        plot_size=(13.5, 9.2),
    )
    beeswarm_figure = plt.gcf()
    beeswarm_axis = beeswarm_figure.axes[0]
    beeswarm_axis.set_facecolor("white")
    beeswarm_axis.set_xlabel(
        "Surrogate SHAP value (impact on hard-class probability)"
        if task == "classification"
        else "Surrogate SHAP value (impact on predicted performance)"
    )
    status_text = (
        "Cross-fitted surrogate TreeSHAP"
        if manuscript_ready
        else "EXPLORATORY: low-fidelity surrogate TreeSHAP"
    )
    beeswarm_axis.set_title(
        f"TabFM Ensemble — {TASK_DISPLAY[task]} SHAP summary\n"
        f"{status_text}; n={n_rows}; cross-fitted R²={overall_fidelity['r2']:.3f}",
        loc="left", pad=18,
    )
    beeswarm_axis.axvline(0.0, color="#333333", linewidth=0.9, zorder=0)
    beeswarm_axis.grid(axis="x", alpha=0.55)
    beeswarm_figure.subplots_adjust(left=0.42, right=0.94, top=0.86, bottom=0.10)
    save_figure(
        beeswarm_figure, artifact_dir / "figure_shap_beeswarm",
        args.figure_formats, args.figure_dpi, plt,
    )

    top = importance.head(int(args.shap_top_features)).sort_values(
        "mean_abs_shap", ascending=True
    )
    bar_height = max(6.4, 0.38 * len(top) + 2.0)
    figure, axis = plt.subplots(figsize=(12.2, bar_height))
    axis.barh(
        top["feature"], top["mean_abs_shap"],
        color="#0072B2", edgecolor="#12405F", linewidth=0.55,
    )
    axis.set_xlabel(
        "Mean |surrogate SHAP value| (probability units)"
        if task == "classification"
        else "Mean |surrogate SHAP value| (performance units)"
    )
    axis.set_title(
        f"TabFM Ensemble — {TASK_DISPLAY[task]} global feature importance\n"
        f"{status_text}; n={n_rows}; cross-fitted R²={overall_fidelity['r2']:.3f}",
        loc="left", pad=14,
    )
    axis.grid(axis="x", alpha=0.55)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", labelsize=8.5, length=0)
    figure.tight_layout()
    save_figure(
        figure, artifact_dir / "figure_shap_mean_abs_bar",
        args.figure_formats, args.figure_dpi, plt,
    )
    print(
        f"  fast SHAP {task}: overall cross-fitted R2="
        f"{overall_fidelity['r2']:.3f}; beeswarm={artifact_dir}",
        flush=True,
    )
    return methodology


def refresh_existing_figures(
    output_dir: Path,
    tasks: list[str],
    models: list[str],
    metric_functions: dict[str, Any],
    pd: Any,
    plt: Any,
    np: Any,
    figure_formats: list[str],
    figure_dpi: int,
) -> None:
    """Rebuild every available figure from persisted, source-backed CSVs."""
    for task in tasks:
        task_dir = output_dir / task
        required = {
            "comparison": task_dir / "model_comparison.csv",
            "predictions": task_dir / "all_models_oof_predictions.csv",
            "folds": task_dir / "all_models_per_fold.csv",
            "paired": task_dir / "paired_differences_vs_tabfm.csv",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Cannot refresh {task} figures; missing: {missing}"
            )
        comparison = pd.read_csv(required["comparison"])
        combined_predictions = pd.read_csv(required["predictions"])
        pilot_level_summary(task, combined_predictions, pd, np).to_csv(
            task_dir / "pilot_summary_by_model.csv", index=False
        )
        combined_folds = pd.read_csv(required["folds"])
        paired = pd.read_csv(required["paired"])
        present_models = [
            model for model in models
            if model in set(combined_predictions["model"].astype(str))
        ]
        predictions_by_model = {
            model: combined_predictions.loc[
                combined_predictions["model"].astype(str).eq(model)
            ].copy()
            for model in present_models
        }
        plot_task_comparison(
            task, comparison.loc[comparison["model"].isin(present_models)],
            task_dir, plt, np, figure_formats, figure_dpi
        )
        plot_paired_advantages(
            task, paired.loc[paired["comparator_model"].isin(present_models)],
            task_dir, plt, np, figure_formats, figure_dpi
        )
        plot_fold_stability(
            task, combined_folds.loc[combined_folds["model"].isin(present_models)],
            task_dir, plt, figure_formats, figure_dpi
        )
        if task == "classification":
            plot_classification_diagnostics(
                predictions_by_model, task_dir, metric_functions, plt, np,
                figure_formats, figure_dpi
            )
            plot_confusion_grid(
                predictions_by_model, task_dir, plt, np,
                figure_formats, figure_dpi
            )
        else:
            plot_regression_diagnostics(
                predictions_by_model, task_dir, plt, np,
                figure_formats, figure_dpi
            )
        if "tabfm_ensemble" in predictions_by_model:
            plot_tabfm_ensemble_diagnostics(
                task,
                predictions_by_model["tabfm_ensemble"],
                task_dir,
                pd,
                plt,
                np,
                figure_formats,
                figure_dpi,
            )
        modality_frames: list[Any] = []
        for model_key in present_models:
            model_dir = task_dir / "models" / model_key
            frequency_path = model_dir / "selected_feature_frequency.csv"
            if frequency_path.is_file():
                feature_frequency = pd.read_csv(frequency_path)
                feature_frequency["modality"] = feature_frequency["feature"].map(assign_modality)
                plot_feature_frequency(
                    feature_frequency, model_key, model_dir, plt,
                    figure_formats, figure_dpi
                )
            selections_path = model_dir / "all_fold_feature_selections.csv"
            if selections_path.is_file():
                selections = pd.read_csv(selections_path)
            else:
                selected_paths = sorted((model_dir / "folds").glob("fold_*_selected_features.csv"))
                frames = []
                for fold_index, path in enumerate(selected_paths, start=1):
                    frame = pd.read_csv(path)
                    frame["fold"] = fold_index
                    frames.append(frame)
                selections = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if not selections.empty:
                selections["selected"] = selections["selected"].astype(str).str.casefold().eq("true")
                selections["modality"] = selections["feature"].map(assign_modality)
                modality_fold = selections.groupby(["fold", "modality"], as_index=False).agg(
                    available_features=("feature", "nunique"), selected_features=("selected", "sum")
                )
                modality_fold["selection_fraction"] = modality_fold["selected_features"] / modality_fold["available_features"]
                modality_frames.append(
                    modality_fold.groupby("modality", as_index=False)
                    .agg(mean_selection_fraction=("selection_fraction", "mean"))
                    .assign(model=model_key)
                )
        if modality_frames:
            plot_modality_selection(
                pd.concat(modality_frames, ignore_index=True), task_dir, plt, np,
                figure_formats, figure_dpi
            )
        print(f"Refreshed publication figures in {task_dir}", flush=True)


def pilot_level_summary(task: str, predictions: Any, pd: Any, np: Any) -> Any:
    """Create descriptive per-pilot cards for downstream training interfaces."""
    records: list[dict[str, Any]] = []
    for (model_key, subject), group in predictions.groupby(
        ["model", "subject"], sort=True
    ):
        record: dict[str, Any] = {
            "task": task,
            "model": model_key,
            "model_display_name": MODEL_DISPLAY.get(str(model_key), str(model_key)),
            "subject": subject,
            "session_runs": int(len(group)),
        }
        if task == "classification":
            hard_truth = group["actual_class"].astype(str).eq("hard")
            record.update(
                {
                    "accuracy": float(
                        group["correct"].astype(str).str.casefold().eq("true").mean()
                    ),
                    "observed_hard_fraction": float(hard_truth.mean()),
                    "mean_predicted_hard_probability": float(
                        group["probability_hard"].mean()
                    ),
                    "mean_absolute_probability_error": float(
                        np.mean(
                            np.abs(
                                group["probability_hard"].to_numpy(float)
                                - hard_truth.to_numpy(float)
                            )
                        )
                    ),
                }
            )
        else:
            residual = (
                group["actual_performance"].to_numpy(float)
                - group["predicted_performance"].to_numpy(float)
            )
            record.update(
                {
                    "observed_performance_mean": float(
                        group["actual_performance"].mean()
                    ),
                    "predicted_performance_mean": float(
                        group["predicted_performance"].mean()
                    ),
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual ** 2))),
                    "mean_residual_observed_minus_predicted": float(
                        np.mean(residual)
                    ),
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    validate_arguments(args)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.feature_selection import (
            SelectKBest,
            f_classif,
            f_regression,
            mutual_info_classif,
            mutual_info_regression,
        )
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            log_loss,
            mean_absolute_error,
            mean_absolute_percentage_error,
            mean_squared_error,
            median_absolute_error,
            r2_score,
            roc_auc_score,
            roc_curve,
        )
        from sklearn.model_selection import (
            GridSearchCV,
            GroupKFold,
            RandomizedSearchCV,
            StratifiedGroupKFold,
            StratifiedKFold,
        )
        from sklearn.neighbors import (
            KNeighborsClassifier,
            KNeighborsRegressor,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}. Install requirements.txt in your active environment."
        ) from exc

    configure_publication_style(plt)
    fast_surrogate_shap = bool(
        args.explain and args.shap_method == "tree_surrogate"
    )
    shap = None
    if args.explain and not args.validate_only and not args.refresh_figures_only:
        try:
            import shap as imported_shap
        except ImportError as exc:
            raise SystemExit(
                "SHAP is required by --explain. Install it in your environment "
                "with: python -m pip install shap"
            ) from exc
        shap = imported_shap

    XGBClassifier = None
    XGBRegressor = None
    if (
        "xgboost" in args.models
        and not args.validate_only
        and not args.refresh_figures_only
    ):
        try:
            from xgboost import XGBClassifier as ImportedXGBClassifier
            from xgboost import XGBRegressor as ImportedXGBRegressor
        except ImportError as exc:
            raise SystemExit(
                "XGBoost is missing. Run: python -m pip install xgboost"
            ) from exc
        XGBClassifier = ImportedXGBClassifier
        XGBRegressor = ImportedXGBRegressor

    tabfm_module = None
    torch = None
    device = args.device
    dtype = args.dtype
    if (
        "tabfm_ensemble" in args.models
        and not args.validate_only
        and not args.refresh_figures_only
        and not fast_surrogate_shap
    ):
        try:
            import tabfm as imported_tabfm
            import torch as imported_torch
        except ImportError as exc:
            raise SystemExit(
                f"TabFM/PyTorch is missing: {exc}. Follow README.md installation instructions."
            ) from exc
        tabfm_module = imported_tabfm
        torch = imported_torch
        device, dtype = resolve_device_dtype(args, torch)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    metric_functions = {
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "f1_score": f1_score,
        "roc_auc_score": roc_auc_score,
        "roc_curve": roc_curve,
        "log_loss": log_loss,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "median_absolute_error": median_absolute_error,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
    }
    splitters = {
        "StratifiedKFold": StratifiedKFold,
        "StratifiedGroupKFold": StratifiedGroupKFold,
        "GroupKFold": GroupKFold,
    }
    optimization_classes = {
        "Pipeline": Pipeline,
        "SelectKBest": SelectKBest,
        "SimpleImputer": SimpleImputer,
        "StandardScaler": StandardScaler,
        "RandomizedSearchCV": RandomizedSearchCV,
        "GridSearchCV": GridSearchCV,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "RandomForestClassifier": RandomForestClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "KNeighborsClassifier": KNeighborsClassifier,
        "KNeighborsRegressor": KNeighborsRegressor,
        "XGBClassifier": XGBClassifier,
        "XGBRegressor": XGBRegressor,
    }

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Workbook not found: {input_path}")
    frame = pd.read_excel(input_path, sheet_name=parse_sheet(args.sheet))
    if frame.empty:
        raise ValueError("The selected worksheet is empty.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")

    subject_column = resolve_column(frame.columns, args.subject_column)
    level_column = resolve_column(frame.columns, args.level_column)
    run_column = resolve_column(frame.columns, args.run_column)
    flight_hours_column = resolve_column(
        frame.columns,
        args.flight_hours_column,
    )
    performance_column = resolve_column(
        frame.columns,
        args.performance_column,
    )
    levels = pd.to_numeric(frame[level_column], errors="raise").to_numpy(
        dtype=int
    )
    unexpected_levels = sorted(set(levels.tolist()) - {1, 2, 3, 4})
    if unexpected_levels:
        raise ValueError(f"Unexpected difficulty levels: {unexpected_levels}")
    performance = pd.to_numeric(
        frame[performance_column],
        errors="raise",
    ).to_numpy(dtype=float)
    if not np.isfinite(performance).all():
        raise ValueError("Performance contains missing or non-finite values.")
    if frame[subject_column].isna().any():
        raise ValueError("Subject contains missing identifiers.")
    subjects = frame[subject_column].to_numpy(copy=True)
    source_excel_rows = np.arange(len(frame), dtype=int) + 2
    resolve_targeted_shap_rows(
        args=args,
        frame=frame,
        subject_column=subject_column,
        run_column=run_column,
        source_excel_rows=source_excel_rows,
        pd=pd,
        np=np,
    )
    if args.shap_source_rows:
        available_source_rows = set(source_excel_rows.astype(int).tolist())
        missing_source_rows = sorted(
            set(int(value) for value in args.shap_source_rows).difference(
                available_source_rows
            )
        )
        if missing_source_rows:
            raise ValueError(
                "Requested --shap-source-rows are outside the workbook: "
                f"{missing_source_rows}"
            )
    binary_difficulty = np.where(levels <= 2, "easy", "hard")

    task_features: dict[str, list[Any]] = {}
    task_splits: dict[str, list[tuple[int, Any, Any]]] = {}
    task_audits: dict[str, Any] = {}
    for task in args.tasks:
        features = feature_columns_for_task(
            task=task,
            feature_set=args.feature_set,
            columns=frame.columns,
            subject_column=subject_column,
            level_column=level_column,
            run_column=run_column,
            flight_hours_column=flight_hours_column,
            performance_column=performance_column,
        )
        if not features:
            raise ValueError(f"No predictors remain for {task}.")
        non_numeric = [
            column
            for column in features
            if not pd.api.types.is_numeric_dtype(frame[column])
        ]
        if non_numeric:
            raise ValueError(
                f"Non-numeric {task} predictors: {non_numeric[:20]}"
            )
        task_features[task] = features
        splits = make_splits(
            task=task,
            protocol=args.split_protocol,
            levels=levels,
            performance=performance,
            subjects=subjects,
            folds=args.folds,
            seed=args.seed,
            splitters=splitters,
            pd=pd,
            np=np,
        )
        task_splits[task] = splits
        audit_records: list[dict[str, Any]] = []
        assigned = np.zeros(len(frame), dtype=int)
        for fold_number, train_indices, test_indices in splits:
            assigned[test_indices] += 1
            train_pilots = set(subjects[train_indices])
            test_pilots = set(subjects[test_indices])
            audit_records.append(
                {
                    "fold": fold_number,
                    "n_train_rows": int(len(train_indices)),
                    "n_test_rows": int(len(test_indices)),
                    "n_train_pilots": int(len(train_pilots)),
                    "n_test_pilots": int(len(test_pilots)),
                    "n_overlapping_pilots": int(
                        len(train_pilots.intersection(test_pilots))
                    ),
                    "test_performance_min": float(
                        np.min(performance[test_indices])
                    ),
                    "test_performance_mean": float(
                        np.mean(performance[test_indices])
                    ),
                    "test_performance_max": float(
                        np.max(performance[test_indices])
                    ),
                    **{
                        f"test_level_{level}": int(
                            np.sum(levels[test_indices] == level)
                        )
                        for level in (1, 2, 3, 4)
                    },
                }
            )
        if not np.all(assigned == 1):
            raise RuntimeError(
                f"{task} folds do not test each row exactly once."
            )
        task_audits[task] = pd.DataFrame(audit_records)

    print(f"Validated {input_path}")
    print(f"  sheet={args.sheet!r}, rows={len(frame)}, columns={len(frame.columns)}")
    print(
        f"  pilots={pd.Series(subjects).nunique()}, "
        f"levels={pd.Series(levels).value_counts().sort_index().to_dict()}"
    )
    print(f"  performance={regression_target_summary(performance, np)}")
    print(f"  feature_set={args.feature_set}, split={args.split_protocol}")
    print(f"  tasks={args.tasks}, models={args.models}")
    print(
        f"  nested optimization: inner_folds={args.inner_folds}, "
        f"selector={args.feature_selector}, "
        f"candidate_features={sorted(args.feature_counts)}, "
        f"classical_random_search_iterations={args.tuning_iterations}"
    )
    for task in args.tasks:
        print(
            f"  {task}: predictors={len(task_features[task])}, "
            f"fold pilot overlap="
            f"{task_audits[task]['n_overlapping_pilots'].tolist()}"
        )
    if args.refresh_figures_only:
        output_dir = args.output_dir.expanduser().resolve()
        refresh_existing_figures(
            output_dir,
            list(args.tasks),
            list(args.models),
            metric_functions,
            pd,
            plt,
            np,
            args.figure_formats,
            args.figure_dpi,
        )
        print("Figure refresh complete; no model was loaded or fitted.")
        return
    if args.validate_only:
        print("Validation-only run complete; no model was fitted.")
        return

    if fast_surrogate_shap:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        summaries: dict[str, Any] = {}
        for task in args.tasks:
            print(
                f"\nFast global SHAP / {TASK_DISPLAY[task]}: reading saved "
                "TabFM OOF predictions; no TabFM checkpoint will be loaded.",
                flush=True,
            )
            summaries[task] = run_fast_surrogate_shap(
                task=task,
                frame=frame,
                task_features=task_features[task],
                source_excel_rows=source_excel_rows,
                output_dir=output_dir,
                args=args,
                shap=shap,
                ExtraTreesRegressor=ExtraTreesRegressor,
                SimpleImputer=SimpleImputer,
                metric_functions=metric_functions,
                pd=pd,
                plt=plt,
                np=np,
            )
        save_json(
            output_dir / "fast_shap_surrogate_summary.json", summaries
        )
        print(
            "Fast-surrogate SHAP complete; TabFM was not loaded or refitted.",
            flush=True,
        )
        return

    checkpoint_root = args.checkpoint_path.expanduser().resolve()
    tabfm_checkpoint_arguments: dict[str, str | None] = {}
    if "tabfm_ensemble" in args.models:
        for requested_task in args.tasks:
            model_type = (
                "classification"
                if requested_task == "classification"
                else "regression"
            )
            tabfm_checkpoint_arguments[requested_task] = checkpoint_argument(
                checkpoint_root,
                model_type,
                args.download_missing_checkpoints,
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.explain and args.shap_method == "fast_tabfm":
        summaries: dict[str, Any] = {}
        for task in args.tasks:
            model_type = (
                "classification" if task == "classification" else "regression"
            )
            print(
                f"\nFast TabFM SHAP / {TASK_DISPLAY[task]}: loading the "
                f"{model_type} backbone once; the costly 32-member NNLS/"
                "calibration fit is skipped.",
                flush=True,
            )
            checkpoint_path = tabfm_checkpoint_arguments[task]
            if args.tabfm_loader == "low_memory":
                backbone = load_tabfm_low_memory(
                    model_type=model_type,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    dtype=dtype,
                    torch=torch,
                )
            else:
                backbone = tabfm_module.tabfm_v1_0_0_pytorch.load(
                    model_type=model_type,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    dtype=dtype,
                    use_cache=False,
                )
            summaries[task] = run_fast_tabfm_shap(
                task=task,
                frame=frame,
                task_features=task_features[task],
                source_excel_rows=source_excel_rows,
                output_dir=output_dir,
                tabfm_backbone=backbone,
                tabfm_module=tabfm_module,
                args=args,
                metric_functions=metric_functions,
                pd=pd,
                plt=plt,
                np=np,
            )
            del backbone
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        save_json(output_dir / "fast_tabfm_shap_summary.json", summaries)
        print(
            "Fast TabFM SHAP complete; the full 32-member ensemble was not "
            "refitted.",
            flush=True,
        )
        return
    all_requested_features = list(
        dict.fromkeys(
            feature for task in args.tasks for feature in task_features[task]
        )
    )
    modality_map = feature_modality_frame(all_requested_features, pd)
    modality_map.to_csv(output_dir / "feature_modality_map.csv", index=False)
    unmapped = modality_map.loc[
        modality_map["modality"].astype(str).eq("Other"), "feature"
    ].tolist()
    if unmapped:
        print(
            "WARNING: modality mapping left features as Other: "
            f"{unmapped[:20]}",
            flush=True,
        )
    plot_dataset_overview(
        levels,
        performance,
        subjects,
        output_dir,
        pd,
        plt,
        np,
        args.figure_formats,
        args.figure_dpi,
    )
    manifest = {
        "script": "tabfm_dual_task_optimized.py",
        "input_file": str(input_path),
        "input_sha256": sha256_file(input_path),
        "sheet": args.sheet,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "subject_column": str(subject_column),
        "level_column": str(level_column),
        "run_column": str(run_column),
        "flight_hours_column": str(flight_hours_column),
        "performance_column": str(performance_column),
        "classification_mapping": {
            "levels_1_2": "easy",
            "levels_3_4": "hard",
        },
        "tasks": list(args.tasks),
        "models": list(args.models),
        "feature_set": args.feature_set,
        "features_by_task": {
            task: [str(column) for column in task_features[task]]
            for task in args.tasks
        },
        "modality_mapping": {
            "artifact": "feature_modality_map.csv",
            "method": "audited deterministic feature-name rules",
            "counts": {
                str(key): int(value)
                for key, value in modality_map["modality"]
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "split_protocol": args.split_protocol,
        "folds": int(args.folds),
        "inner_folds": int(args.inner_folds),
        "seed": int(args.seed),
        "feature_selector": args.feature_selector,
        "feature_counts": sorted(int(value) for value in args.feature_counts),
        "tuning_iterations": int(args.tuning_iterations),
        "selector_proxy_trees": int(args.selector_proxy_trees),
        "tabfm_estimators": int(args.tabfm_estimators),
        "tabfm_batch_size": int(args.tabfm_batch_size),
        "tabfm_loader": args.tabfm_loader,
        "xgb_trees": int(args.xgb_trees),
        "rf_trees": int(args.rf_trees),
        "knn_neighbors": int(args.knn_neighbors),
        "bootstrap_samples": int(args.bootstrap_samples),
        "confidence_level": float(args.confidence_level),
        "bootstrap_seed": int(args.bootstrap_seed),
        "explain": bool(args.explain),
        "shap_method": args.shap_method if args.explain else None,
        "shap_background_size": int(args.shap_background_size),
        "shap_explain_rows": int(args.shap_explain_rows),
        "shap_max_evals": int(args.shap_max_evals),
        "shap_checkpoint_rows": int(args.shap_checkpoint_rows),
        "explain_models": list(args.explain_models),
        "shap_source_rows": (
            list(args.shap_source_rows) if args.shap_source_rows else None
        ),
        "shap_subject": args.shap_subject,
        "shap_runs": list(args.shap_runs) if args.shap_runs else None,
        "figure_dpi": int(args.figure_dpi),
        "figure_formats": list(args.figure_formats),
        "requested_device": args.device,
        "requested_dtype": args.dtype,
        "resolved_device": str(device),
        "resolved_dtype": str(dtype),
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists() and args.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        protected_keys = (
            "input_sha256",
            "sheet",
            "tasks",
            "models",
            "feature_set",
            "split_protocol",
            "folds",
            "inner_folds",
            "seed",
            "feature_selector",
            "feature_counts",
            "tuning_iterations",
            "selector_proxy_trees",
            "tabfm_estimators",
            "tabfm_batch_size",
            "tabfm_loader",
            "xgb_trees",
            "rf_trees",
            "knn_neighbors",
            "requested_device",
            "requested_dtype",
        )
        mismatches = [
            key
            for key in protected_keys
            if previous.get(key) != manifest.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "The output directory contains an incompatible resumable "
                f"run. Different settings: {mismatches}. Choose a new "
                "--output-dir."
            )
    save_json(manifest_path, manifest)

    all_task_summaries: dict[str, Any] = {}
    for task in args.tasks:
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        task_audits[task].to_csv(task_dir / "fold_audit.csv", index=False)
        fold_assignment = np.zeros(len(frame), dtype=int)
        for fold_number, _, test_indices in task_splits[task]:
            fold_assignment[test_indices] = fold_number
        pd.DataFrame(
            {
                "source_excel_row": source_excel_rows,
                "subject": subjects,
                "level": levels,
                "difficulty_binary": binary_difficulty,
                "performance": performance,
                "test_fold": fold_assignment,
            }
        ).to_csv(task_dir / "fold_assignments.csv", index=False)

        X = frame[task_features[task]].replace([np.inf, -np.inf], np.nan)
        y_task = binary_difficulty if task == "classification" else performance
        model_fold_paths: dict[str, list[tuple[int, Any, Any, Path]]] = {}
        tabfm_backbone = None

        for model_key in args.models:
            model_dir = task_dir / "models" / model_key
            folds_dir = model_dir / "folds"
            folds_dir.mkdir(parents=True, exist_ok=True)
            paths: list[tuple[int, Any, Any, Path]] = []
            missing: list[tuple[int, Any, Any, Path]] = []
            for fold_number, train_indices, test_indices in task_splits[task]:
                fold_path = folds_dir / f"fold_{fold_number:02d}_predictions.csv"
                paths.append((fold_number, train_indices, test_indices, fold_path))
                expected_rows = set(
                    source_excel_rows[test_indices].astype(int).tolist()
                )
                shap_path = fold_path.with_name(
                    fold_path.name.replace(
                        "_predictions.csv", "_shap_values.csv"
                    )
                )
                explanation_required = (
                    args.explain
                    and model_key in args.explain_models
                    and (
                        not args.shap_source_rows
                        or bool(
                            expected_rows.intersection(args.shap_source_rows)
                        )
                    )
                )
                if not (
                    args.resume
                    and validate_saved_fold(
                        fold_path,
                        model_key,
                        fold_number,
                        expected_rows,
                        pd,
                    )
                    and (
                        not explanation_required
                        or validate_saved_shap(
                            shap_path,
                            model_key,
                            fold_number,
                            expected_rows,
                            args,
                            pd,
                        )
                    )
                ):
                    missing.append(
                        (fold_number, train_indices, test_indices, fold_path)
                    )
            model_fold_paths[model_key] = paths
            print(
                f"\n{TASK_DISPLAY[task]} / {MODEL_DISPLAY[model_key]}: "
                f"{len(paths) - len(missing)}/{len(paths)} folds complete."
            )

            if model_key == "tabfm_ensemble" and missing and tabfm_backbone is None:
                model_type = (
                    "classification" if task == "classification" else "regression"
                )
                checkpoint_path = tabfm_checkpoint_arguments[task]
                print(
                    f"Loading TabFM {model_type} checkpoint on {device} "
                    f"with {dtype}..."
                )
                if args.tabfm_loader == "low_memory":
                    tabfm_backbone = load_tabfm_low_memory(
                        model_type=model_type,
                        checkpoint_path=checkpoint_path,
                        device=device,
                        dtype=dtype,
                        torch=torch,
                    )
                else:
                    tabfm_backbone = tabfm_module.tabfm_v1_0_0_pytorch.load(
                        model_type=model_type,
                        checkpoint_path=checkpoint_path,
                        device=device,
                        dtype=dtype,
                        use_cache=False,
                    )

            for fold_number, train_indices, test_indices, fold_path in missing:
                fold_started = time.monotonic()
                fold_source_rows = set(
                    source_excel_rows[test_indices].astype(int).tolist()
                )
                explain_this_fold = (
                    args.explain
                    and model_key in args.explain_models
                    and (
                        not args.shap_source_rows
                        or bool(
                            fold_source_rows.intersection(
                                args.shap_source_rows
                            )
                        )
                    )
                )
                X_train_raw = X.iloc[train_indices].copy()
                X_test_raw = X.iloc[test_indices].copy()
                usable_columns = [
                    column
                    for column in X.columns
                    if not X_train_raw[column].isna().all()
                ]
                if not usable_columns:
                    raise RuntimeError(
                        f"No usable predictors remain in outer fold {fold_number}."
                    )
                X_train_raw = X_train_raw[usable_columns]
                X_test_raw = X_test_raw[usable_columns]
                y_train = y_task[train_indices]
                y_test = y_task[test_indices]
                cached_state_path = fold_path.with_name(
                    fold_path.name.replace(
                        "_predictions.csv", "_tabfm_fold_state.joblib"
                    )
                )
                selected_audit_path = fold_path.with_name(
                    fold_path.name.replace(
                        "_predictions.csv", "_selected_features.csv"
                    )
                )
                can_reuse_exact_tabfm_state = (
                    model_key == "tabfm_ensemble"
                    and explain_this_fold
                    and args.shap_method == "permutation"
                    and args.resume
                    and cached_state_path.is_file()
                    and selected_audit_path.is_file()
                    and validate_saved_fold(
                        fold_path,
                        model_key,
                        fold_number,
                        fold_source_rows,
                        pd,
                    )
                )
                if can_reuse_exact_tabfm_state:
                    cached_audit = pd.read_csv(selected_audit_path)
                    cached_required = {
                        "feature", "selected", "training_imputation_median"
                    }
                    cached_missing = cached_required.difference(
                        cached_audit.columns
                    )
                    if cached_missing:
                        raise RuntimeError(
                            f"Cached feature audit lacks "
                            f"{sorted(cached_missing)}: {selected_audit_path}"
                        )
                    cached_selected_mask = (
                        cached_audit["selected"].astype(str).str.strip()
                        .str.casefold().isin({"true", "1", "yes"})
                    )
                    cached_tuning_path = fold_path.with_name(
                        fold_path.name.replace("_predictions.csv", "_tuning.json")
                    )
                    cached_selected = json.loads(
                        cached_tuning_path.read_text(encoding="utf-8")
                    )["selected_features"]
                    cached_selected_audit = cached_audit.loc[cached_selected_mask].set_index("feature")
                    if set(cached_selected_audit.index) != set(cached_selected):
                        raise ValueError("Cached tuning and feature audit disagree")
                    cached_medians = pd.to_numeric(
                        cached_selected_audit.loc[cached_selected, "training_imputation_median"],
                        errors="raise",
                    )
                    unavailable = [
                        feature for feature in cached_selected
                        if feature not in X_train_raw.columns
                    ]
                    if unavailable:
                        raise RuntimeError(
                            f"Cached TabFM features are unavailable: "
                            f"{unavailable[:20]}"
                        )
                    train_input = (
                        X_train_raw[cached_selected]
                        .fillna(cached_medians)
                        .reset_index(drop=True)
                    )
                    test_input = (
                        X_test_raw[cached_selected]
                        .fillna(cached_medians)
                        .reset_index(drop=True)
                    )
                    estimator = load_tabfm_fold_state(
                        cached_state_path, tabfm_backbone, joblib
                    )
                    shap_path = fold_path.with_name(
                        fold_path.name.replace(
                            "_predictions.csv", "_shap_values.csv"
                        )
                    )
                    print(
                        f"  reusing exact fitted TabFM fold state: "
                        f"{cached_state_path.name}",
                        flush=True,
                    )
                    with progress_heartbeat(
                        f"cached permutation SHAP {MODEL_DISPLAY[model_key]} "
                        f"fold {fold_number:02d}",
                        args.progress_interval_minutes,
                    ):
                        explain_fold_with_shap(
                            task=task,
                            model_key=model_key,
                            fold_number=fold_number,
                            estimator=estimator,
                            train_input=train_input,
                            test_input=test_input,
                            test_indices=test_indices,
                            source_excel_rows=source_excel_rows,
                            subjects=subjects,
                            output_path=shap_path,
                            args=args,
                            shap=shap,
                            pd=pd,
                            np=np,
                        )
                    del estimator
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                inner_cv = make_inner_cv(
                    task=task,
                    protocol=args.split_protocol,
                    levels=levels[train_indices],
                    performance=performance[train_indices],
                    subjects=subjects[train_indices],
                    folds=args.inner_folds,
                    seed=args.seed + fold_number,
                    splitters=splitters,
                    pd=pd,
                    np=np,
                )
                counts = candidate_feature_counts(
                    args.feature_counts,
                    len(usable_columns),
                )
                score_function = make_selector_score_function(
                    task=task,
                    selector_name=args.feature_selector,
                    seed=args.seed + fold_number,
                    mutual_info_classif=mutual_info_classif,
                    mutual_info_regression=mutual_info_regression,
                    f_classif=f_classif,
                    f_regression=f_regression,
                )
                search_target = encoded_search_target(task, y_train, np)
                if model_key == "tabfm_ensemble":
                    search = build_tabfm_feature_search(
                        task=task,
                        feature_counts=counts,
                        inner_cv=inner_cv,
                        args=args,
                        score_function=score_function,
                        classes=optimization_classes,
                    )
                    selection_strategy = "nested_extra_trees_proxy"
                    search_label = (
                        f"TabFM feature-count search fold {fold_number:02d}"
                    )
                else:
                    search = build_classical_search(
                        model_key=model_key,
                        task=task,
                        feature_counts=counts,
                        inner_cv=inner_cv,
                        args=args,
                        score_function=score_function,
                        classes=optimization_classes,
                    )
                    selection_strategy = "nested_joint_model_search"
                    search_label = (
                        f"{MODEL_DISPLAY[model_key]} nested tuning fold "
                        f"{fold_number:02d}"
                    )

                search_started = time.monotonic()
                with progress_heartbeat(
                    search_label,
                    args.progress_interval_minutes,
                ):
                    search.fit(X_train_raw, search_target)
                search_seconds = time.monotonic() - search_started

                best_pipeline = search.best_estimator_
                feature_audit = feature_audit_from_pipeline(
                    best_pipeline,
                    usable_columns,
                    pd,
                    np,
                )
                selector_support = np.asarray(
                    best_pipeline.named_steps["selector"].get_support(),
                    dtype=bool,
                )
                selected_features = [
                    column
                    for column, selected in zip(usable_columns, selector_support)
                    if selected
                ]
                if not selected_features:
                    raise RuntimeError(
                        f"Feature selection chose no predictors in fold {fold_number}."
                    )
                tuning_results = compact_search_results(search, pd)
                tuning_path = fold_path.with_name(
                    fold_path.name.replace(
                        "_predictions.csv",
                        "_tuning.json",
                    )
                )
                selected_path = fold_path.with_name(
                    fold_path.name.replace(
                        "_predictions.csv",
                        "_selected_features.csv",
                    )
                )
                tuning_results_path = fold_path.with_name(
                    fold_path.name.replace(
                        "_predictions.csv",
                        "_tuning_results.csv",
                    )
                )
                inner_score = float(search.best_score_)
                save_json(
                    tuning_path,
                    {
                        "task": task,
                        "model": model_key,
                        "outer_fold": int(fold_number),
                        "selection_strategy": selection_strategy,
                        "feature_selector": args.feature_selector,
                        "candidate_feature_counts": counts,
                        "selected_feature_count": int(len(selected_features)),
                        "selected_features": [
                            str(feature) for feature in selected_features
                        ],
                        "inner_folds": int(args.inner_folds),
                        "scoring": search_scoring(task),
                        "best_inner_cv_score": inner_score,
                        "best_inner_cv_primary_metric": (
                            inner_score if task == "classification" else -inner_score
                        ),
                        "best_params": search.best_params_,
                        "candidates_evaluated": int(len(tuning_results)),
                    },
                )
                feature_audit.to_csv(selected_path, index=False)
                tuning_results.to_csv(tuning_results_path, index=False)
                print(
                    f"  selected {len(selected_features)}/{len(usable_columns)} "
                    f"predictors | best inner score={inner_score:.5f}",
                    flush=True,
                )
                probability_deviation = None
                final_fit_seconds = 0.0
                prediction_seconds = 0.0
                ensemble_diagnostics: dict[str, Any] = {}

                if task == "classification":
                    if model_key == "tabfm_ensemble":
                        imputed_train = best_pipeline.named_steps[
                            "imputer"
                        ].transform(X_train_raw)
                        imputed_test = best_pipeline.named_steps[
                            "imputer"
                        ].transform(X_test_raw)
                        train_array = best_pipeline.named_steps[
                            "selector"
                        ].transform(imputed_train)
                        test_array = best_pipeline.named_steps[
                            "selector"
                        ].transform(imputed_test)
                        estimator = tabfm_module.TabFMClassifier.ensemble(
                            model=tabfm_backbone,
                            n_estimators=args.tabfm_estimators,
                            max_num_features=500,
                            max_num_rows=None,
                            batch_size=args.tabfm_batch_size,
                            random_state=args.seed,
                            verbose=False,
                        )
                        train_input = pd.DataFrame(
                            train_array,
                            columns=selected_features,
                        )
                        test_input = pd.DataFrame(
                            test_array,
                            columns=selected_features,
                        )
                        with progress_heartbeat(
                            f"TabFM classification fold {fold_number:02d} "
                            f"({args.tabfm_estimators} estimators)",
                            args.progress_interval_minutes,
                        ):
                            fit_started = time.monotonic()
                            estimator.fit(train_input, y_train)
                            final_fit_seconds = time.monotonic() - fit_started
                        save_tabfm_fold_state(
                            estimator,
                            fold_path.with_name(
                                fold_path.name.replace(
                                    "_predictions.csv",
                                    "_tabfm_fold_state.joblib",
                                )
                            ),
                            joblib,
                        )
                        prediction_started = time.monotonic()
                        with progress_heartbeat(
                            f"TabFM classification prediction fold {fold_number:02d}",
                            args.progress_interval_minutes,
                        ):
                            member_logits = estimator._predict_proba_internal(
                                test_input
                            )
                            probabilities, probability_deviation = (
                                align_binary_probabilities(
                                    estimator._process_logits(member_logits),
                                    estimator.classes_,
                                    np,
                                )
                            )
                            member_probabilities = estimator.softmax(
                                member_logits,
                                axis=-1,
                                temperature=estimator.softmax_temperature,
                            )
                            hard_index = int(
                                np.flatnonzero(
                                    np.asarray(estimator.classes_) == "hard"
                                )[0]
                            )
                            member_hard = member_probabilities[
                                :, :, hard_index
                            ]
                            ensemble_diagnostics = {
                                "ensemble_member_probability_hard_mean_uncalibrated": np.mean(
                                    member_hard, axis=0
                                ),
                                "ensemble_member_probability_hard_std_uncalibrated": np.std(
                                    member_hard, axis=0, ddof=1
                                ),
                                "ensemble_member_probability_hard_min_uncalibrated": np.min(
                                    member_hard, axis=0
                                ),
                                "ensemble_member_probability_hard_max_uncalibrated": np.max(
                                    member_hard, axis=0
                                ),
                            }
                        prediction_seconds = time.monotonic() - prediction_started
                    else:
                        estimator = best_pipeline
                        prediction_started = time.monotonic()
                        probabilities, probability_deviation = (
                            align_binary_probabilities(
                                estimator.predict_proba(X_test_raw),
                                np.asarray(["easy", "hard"]),
                                np,
                            )
                        )
                        prediction_seconds = time.monotonic() - prediction_started
                    predictions = np.asarray(["easy", "hard"])[
                        np.argmax(probabilities, axis=1)
                    ]
                    prediction_data = common_prediction_columns(
                        frame,
                        test_indices,
                        source_excel_rows,
                        model_key,
                        fold_number,
                        subject_column,
                        level_column,
                        run_column,
                        flight_hours_column,
                        pd,
                    )
                    prediction_data.update(
                        {
                            "actual_class": y_test,
                            "predicted_class": predictions,
                            "probability_easy": probabilities[:, 0],
                            "probability_hard": probabilities[:, 1],
                            "correct": predictions == y_test,
                            **ensemble_diagnostics,
                        }
                    )
                else:
                    if model_key == "tabfm_ensemble":
                        imputed_train = best_pipeline.named_steps[
                            "imputer"
                        ].transform(X_train_raw)
                        imputed_test = best_pipeline.named_steps[
                            "imputer"
                        ].transform(X_test_raw)
                        train_array = best_pipeline.named_steps[
                            "selector"
                        ].transform(imputed_train)
                        test_array = best_pipeline.named_steps[
                            "selector"
                        ].transform(imputed_test)
                        estimator = tabfm_module.TabFMRegressor.ensemble(
                            model=tabfm_backbone,
                            n_estimators=args.tabfm_estimators,
                            max_num_features=500,
                            max_num_rows=None,
                            batch_size=args.tabfm_batch_size,
                            random_state=args.seed,
                            verbose=False,
                        )
                        train_input = pd.DataFrame(
                            train_array,
                            columns=selected_features,
                        )
                        test_input = pd.DataFrame(
                            test_array,
                            columns=selected_features,
                        )
                        with progress_heartbeat(
                            f"TabFM regression fold {fold_number:02d} "
                            f"({args.tabfm_estimators} estimators)",
                            args.progress_interval_minutes,
                        ):
                            fit_started = time.monotonic()
                            estimator.fit(train_input, y_train)
                            final_fit_seconds = time.monotonic() - fit_started
                        save_tabfm_fold_state(
                            estimator,
                            fold_path.with_name(
                                fold_path.name.replace(
                                    "_predictions.csv",
                                    "_tabfm_fold_state.joblib",
                                )
                            ),
                            joblib,
                        )
                        prediction_started = time.monotonic()
                        with progress_heartbeat(
                            f"TabFM regression prediction fold {fold_number:02d}",
                            args.progress_interval_minutes,
                        ):
                            member_predictions_scaled = (
                                estimator._predict_internal(test_input)
                            )
                            predictions = np.asarray(
                                estimator._combine_predictions(
                                    member_predictions_scaled
                                ),
                                dtype=float,
                            ).reshape(-1)
                            member_predictions = np.vstack(
                                [
                                    np.asarray(
                                        estimator._inverse_transform_y(
                                            member_predictions_scaled[index]
                                        ),
                                        dtype=float,
                                    ).reshape(-1)
                                    for index in range(
                                        member_predictions_scaled.shape[0]
                                    )
                                ]
                            )
                            ensemble_diagnostics = {
                                "ensemble_member_prediction_mean": np.mean(
                                    member_predictions, axis=0
                                ),
                                "ensemble_member_prediction_std": np.std(
                                    member_predictions, axis=0, ddof=1
                                ),
                                "ensemble_member_prediction_min": np.min(
                                    member_predictions, axis=0
                                ),
                                "ensemble_member_prediction_max": np.max(
                                    member_predictions, axis=0
                                ),
                            }
                        prediction_seconds = time.monotonic() - prediction_started
                    else:
                        estimator = best_pipeline
                        prediction_started = time.monotonic()
                        predictions = np.asarray(
                            estimator.predict(X_test_raw),
                            dtype=float,
                        )
                        prediction_seconds = time.monotonic() - prediction_started
                    if not np.isfinite(predictions).all():
                        raise RuntimeError(
                            f"{MODEL_DISPLAY[model_key]} returned non-finite "
                            "regression predictions."
                        )
                    prediction_data = common_prediction_columns(
                        frame,
                        test_indices,
                        source_excel_rows,
                        model_key,
                        fold_number,
                        subject_column,
                        level_column,
                        run_column,
                        flight_hours_column,
                        pd,
                    )
                    prediction_data.update(
                        {
                            "actual_performance": y_test,
                            "predicted_performance": predictions,
                            "residual": y_test - predictions,
                            "absolute_error": np.abs(y_test - predictions),
                            **ensemble_diagnostics,
                        }
                    )

                if explain_this_fold:
                    shap_path = fold_path.with_name(
                        fold_path.name.replace(
                            "_predictions.csv", "_shap_values.csv"
                        )
                    )
                    if model_key == "tabfm_ensemble":
                        shap_train_input = train_input
                        shap_test_input = test_input
                    else:
                        shap_train_input = transformed_selected_inputs(
                            best_pipeline, X_train_raw, selected_features, pd
                        )
                        shap_test_input = transformed_selected_inputs(
                            best_pipeline, X_test_raw, selected_features, pd
                        )
                    with progress_heartbeat(
                        f"permutation SHAP {MODEL_DISPLAY[model_key]} fold {fold_number:02d}",
                        args.progress_interval_minutes,
                    ):
                        explain_fold_with_shap(
                            task=task,
                            model_key=model_key,
                            fold_number=fold_number,
                            estimator=estimator,
                            train_input=shap_train_input,
                            test_input=shap_test_input,
                            test_indices=test_indices,
                            source_excel_rows=source_excel_rows,
                            subjects=subjects,
                            output_path=shap_path,
                            args=args,
                            shap=shap,
                            pd=pd,
                            np=np,
                        )

                fold_predictions = pd.DataFrame(prediction_data).sort_values(
                    "source_excel_row"
                )
                fold_metrics = compute_metrics(
                    task,
                    fold_predictions,
                    metric_functions,
                    np,
                )
                fold_metrics.update(
                    {
                        "task": task,
                        "model": model_key,
                        "fold": fold_number,
                        "n_train_rows": int(len(train_indices)),
                        "n_test_rows": int(len(test_indices)),
                        "n_train_pilots": int(
                            len(set(subjects[train_indices]))
                        ),
                        "n_test_pilots": int(len(set(subjects[test_indices]))),
                        "n_overlapping_pilots": int(
                            len(
                                set(subjects[train_indices]).intersection(
                                    set(subjects[test_indices])
                                )
                            )
                        ),
                        "n_candidate_predictors": int(len(usable_columns)),
                        "n_predictors": int(len(selected_features)),
                        "feature_selector": args.feature_selector,
                        "selection_strategy": selection_strategy,
                        "inner_cv_scoring": search_scoring(task),
                        "best_inner_cv_score": inner_score,
                        "probability_sum_max_deviation": probability_deviation,
                        "search_seconds": float(search_seconds),
                        "final_fit_seconds": float(final_fit_seconds),
                        "prediction_seconds": float(prediction_seconds),
                        "prediction_milliseconds_per_row": float(
                            1000.0 * prediction_seconds / len(test_indices)
                        ),
                        "total_fold_seconds": float(
                            time.monotonic() - fold_started
                        ),
                    }
                )
                fold_predictions.to_csv(fold_path, index=False)
                save_json(
                    fold_path.with_name(
                        fold_path.name.replace(
                            "_predictions.csv",
                            "_metrics.json",
                        )
                    ),
                    fold_metrics,
                )
                feature_audit.to_csv(
                    fold_path.with_name(
                        fold_path.name.replace(
                            "_predictions.csv",
                            "_preprocessing.csv",
                        )
                    ),
                    index=False,
                )
                metric_text = ", ".join(
                    f"{key}={value:.4f}"
                    for key, value in fold_metrics.items()
                    if key in ({"accuracy", "roc_auc_hard"} if task == "classification" else {"rmse", "r2"})
                )
                print(f"  fold {fold_number:02d} saved | {metric_text}")
                del estimator, search, best_pipeline, feature_audit, tuning_results
                if model_key == "tabfm_ensemble":
                    del train_array, test_array, imputed_train, imputed_test
                gc.collect()
                if torch is not None and device == "cuda":
                    torch.cuda.empty_cache()

        predictions_by_model: dict[str, Any] = {}
        metrics_by_model: dict[str, dict[str, float]] = {}
        stability_by_model: dict[str, Any] = {}
        draws_by_model: dict[str, Any] = {}
        folds_by_model: dict[str, Any] = {}
        feature_counts_by_model: dict[str, dict[str, float]] = {}
        modality_by_model: dict[str, Any] = {}
        shap_modality_frames: list[Any] = []
        for model_key in args.models:
            model_dir = task_dir / "models" / model_key
            fold_frames = [
                pd.read_csv(path)
                for _, _, _, path in model_fold_paths[model_key]
            ]
            predictions = (
                pd.concat(fold_frames, ignore_index=True)
                .sort_values("source_excel_row")
                .reset_index(drop=True)
            )
            validate_oof(predictions, source_excel_rows, model_key)
            overall = compute_metrics(
                task,
                predictions,
                metric_functions,
                np,
            )
            fold_records: list[dict[str, Any]] = []
            for fold_number, _, _, path in model_fold_paths[model_key]:
                fold_frame = pd.read_csv(path)
                record = compute_metrics(
                    task,
                    fold_frame,
                    metric_functions,
                    np,
                )
                record.update(
                    {
                        "task": task,
                        "model": model_key,
                        "model_display_name": MODEL_DISPLAY[model_key],
                        "fold": fold_number,
                        "n_test_rows": int(len(fold_frame)),
                        "n_test_pilots": int(fold_frame["subject"].nunique()),
                    }
                )
                saved_metrics_path = path.with_name(
                    path.name.replace("_predictions.csv", "_metrics.json")
                )
                if saved_metrics_path.is_file():
                    saved_fold_metrics = json.loads(
                        saved_metrics_path.read_text(encoding="utf-8")
                    )
                    for key in (
                        "n_candidate_predictors",
                        "n_predictors",
                        "feature_selector",
                        "selection_strategy",
                        "inner_cv_scoring",
                        "best_inner_cv_score",
                        "search_seconds",
                        "final_fit_seconds",
                        "prediction_seconds",
                        "prediction_milliseconds_per_row",
                        "total_fold_seconds",
                    ):
                        record[key] = saved_fold_metrics.get(key)
                fold_records.append(record)
            fold_metrics = pd.DataFrame(fold_records).sort_values("fold")
            selection_frames: list[Any] = []
            tuning_records: list[dict[str, Any]] = []
            for fold_number, _, _, path in model_fold_paths[model_key]:
                selected_path = path.with_name(
                    path.name.replace(
                        "_predictions.csv",
                        "_selected_features.csv",
                    )
                )
                if not selected_path.is_file():
                    raise RuntimeError(
                        f"Missing selected-feature audit: {selected_path}"
                    )
                selected_frame = pd.read_csv(selected_path)
                selected_frame["selected"] = (
                    selected_frame["selected"]
                    .astype(str)
                    .str.casefold()
                    .eq("true")
                )
                selected_frame["modality"] = selected_frame["feature"].map(
                    assign_modality
                )
                selected_frame["fold"] = int(fold_number)
                selection_frames.append(selected_frame)
                tuning_path = path.with_name(
                    path.name.replace("_predictions.csv", "_tuning.json")
                )
                if not tuning_path.is_file():
                    raise RuntimeError(f"Missing tuning audit: {tuning_path}")
                tuning_payload = json.loads(
                    tuning_path.read_text(encoding="utf-8")
                )
                tuning_records.append(
                    {
                        "task": task,
                        "model": model_key,
                        "fold": int(fold_number),
                        "selection_strategy": tuning_payload[
                            "selection_strategy"
                        ],
                        "feature_selector": tuning_payload["feature_selector"],
                        "selected_feature_count": int(
                            tuning_payload["selected_feature_count"]
                        ),
                        "scoring": tuning_payload["scoring"],
                        "best_inner_cv_score": float(
                            tuning_payload["best_inner_cv_score"]
                        ),
                        "best_inner_cv_primary_metric": float(
                            tuning_payload["best_inner_cv_primary_metric"]
                        ),
                        "best_params_json": json.dumps(
                            tuning_payload["best_params"],
                            sort_keys=True,
                        ),
                    }
                )
            all_selections = pd.concat(selection_frames, ignore_index=True)
            selected_counts = (
                all_selections.groupby("fold")["selected"].sum().astype(int)
            )
            feature_frequency = (
                all_selections.groupby(["feature", "modality"], as_index=False)
                .agg(
                    selected_folds=("selected", "sum"),
                    selection_frequency=("selected", "mean"),
                    mean_selection_score=("selection_score", "mean"),
                    mean_selection_rank=("selection_rank", "mean"),
                )
                .sort_values(
                    ["selected_folds", "mean_selection_rank"],
                    ascending=[False, True],
                )
            )
            feature_frequency.to_csv(
                model_dir / "selected_feature_frequency.csv",
                index=False,
            )
            all_selections.to_csv(
                model_dir / "all_fold_feature_selections.csv",
                index=False,
            )
            modality_fold = (
                all_selections.groupby(["fold", "modality"], as_index=False)
                .agg(
                    available_features=("feature", "nunique"),
                    selected_features=("selected", "sum"),
                )
            )
            modality_fold["selection_fraction"] = (
                modality_fold["selected_features"]
                / modality_fold["available_features"]
            )
            modality_frequency = (
                modality_fold.groupby("modality", as_index=False)
                .agg(
                    mean_selection_fraction=("selection_fraction", "mean"),
                    minimum_selection_fraction=("selection_fraction", "min"),
                    maximum_selection_fraction=("selection_fraction", "max"),
                    mean_selected_features=("selected_features", "mean"),
                )
            )
            modality_fold.to_csv(
                model_dir / "modality_selection_by_fold.csv", index=False
            )
            modality_frequency.to_csv(
                model_dir / "modality_selection_frequency.csv", index=False
            )
            modality_by_model[model_key] = modality_frequency.assign(
                model=model_key,
                model_display_name=MODEL_DISPLAY[model_key],
            )
            plot_feature_frequency(
                feature_frequency,
                model_key,
                model_dir,
                plt,
                args.figure_formats,
                args.figure_dpi,
            )
            pd.DataFrame(tuning_records).sort_values("fold").to_csv(
                model_dir / "tuning_summary.csv",
                index=False,
            )
            feature_count_summary = {
                "selected_predictors_min": int(selected_counts.min()),
                "selected_predictors_mean": float(selected_counts.mean()),
                "selected_predictors_max": int(selected_counts.max()),
                "original_predictors": int(len(task_features[task])),
                "mean_reduction_percent": float(
                    100.0
                    * (1.0 - selected_counts.mean() / len(task_features[task]))
                ),
            }
            stability, draws = cluster_bootstrap(
                task=task,
                predictions_frame=predictions,
                point_metrics=overall,
                metric_functions=metric_functions,
                n_resamples=args.bootstrap_samples,
                confidence_level=args.confidence_level,
                seed=args.bootstrap_seed,
                pd=pd,
                np=np,
            )
            overall_payload: dict[str, Any] = {
                **overall,
                "task": task,
                "model": model_key,
                "model_display_name": MODEL_DISPLAY[model_key],
                "n_rows": int(len(predictions)),
                "n_pilots": int(predictions["subject"].nunique()),
                **feature_count_summary,
                "feature_set": args.feature_set,
                "feature_selector": args.feature_selector,
                "inner_folds": int(args.inner_folds),
                "split_protocol": args.split_protocol,
                "folds": int(args.folds),
                "confidence_intervals": format_ci_payload(stability),
            }
            predictions.to_csv(model_dir / "oof_predictions.csv", index=False)
            predictions.to_excel(model_dir / "oof_predictions.xlsx", index=False)
            fold_metrics.to_csv(model_dir / "per_fold_metrics.csv", index=False)
            stability.to_csv(model_dir / "metric_stability.csv", index=False)
            draws.to_csv(model_dir / "cluster_bootstrap_draws.csv", index=False)
            save_json(model_dir / "overall_metrics.json", overall_payload)

            if args.explain:
                shap_modality = aggregate_shap_outputs(
                    model_dir,
                    model_key,
                    pd,
                    plt,
                    args.figure_formats,
                    args.figure_dpi,
                )
                if shap_modality is not None:
                    shap_modality_frames.append(shap_modality)

            if task == "classification":
                report = classification_report(
                    predictions["actual_class"],
                    predictions["predicted_class"],
                    labels=["easy", "hard"],
                    output_dict=True,
                    zero_division=0,
                )
                save_json(model_dir / "classification_report.json", report)
                pd.DataFrame(report).transpose().to_csv(
                    model_dir / "classification_report.csv"
                )
                confusion = confusion_matrix(
                    predictions["actual_class"],
                    predictions["predicted_class"],
                    labels=["easy", "hard"],
                )
                pd.DataFrame(
                    confusion,
                    index=["actual_easy", "actual_hard"],
                    columns=["predicted_easy", "predicted_hard"],
                ).to_csv(model_dir / "confusion_matrix.csv")
                figure, axis = plt.subplots(figsize=(5.5, 4.5))
                ConfusionMatrixDisplay(
                    confusion_matrix=confusion,
                    display_labels=["easy", "hard"],
                ).plot(
                    ax=axis,
                    cmap="Blues",
                    colorbar=False,
                    values_format="d",
                )
                axis.set_title(MODEL_DISPLAY[model_key])
                figure.tight_layout()
                save_figure(
                    figure,
                    model_dir / "figure_confusion_matrix",
                    args.figure_formats,
                    args.figure_dpi,
                    plt,
                )
            else:
                figure, axis = plt.subplots(figsize=(6, 5.5))
                axis.scatter(
                    predictions["actual_performance"],
                    predictions["predicted_performance"],
                    alpha=0.6,
                    s=24,
                    color=MODEL_COLORS.get(model_key, "#555555"),
                    edgecolors="none",
                    rasterized=True,
                )
                lower = float(
                    min(
                        predictions["actual_performance"].min(),
                        predictions["predicted_performance"].min(),
                    )
                )
                upper = float(
                    max(
                        predictions["actual_performance"].max(),
                        predictions["predicted_performance"].max(),
                    )
                )
                axis.plot([lower, upper], [lower, upper], "--", color="black")
                axis.set_xlabel("True performance")
                axis.set_ylabel("Predicted performance")
                axis.set_title(MODEL_DISPLAY[model_key])
                axis.grid(alpha=0.55)
                axis.spines[["top", "right"]].set_visible(False)
                figure.tight_layout()
                save_figure(
                    figure,
                    model_dir / "figure_actual_vs_predicted",
                    args.figure_formats,
                    args.figure_dpi,
                    plt,
                )

            predictions_by_model[model_key] = predictions
            metrics_by_model[model_key] = overall
            stability_by_model[model_key] = stability
            draws_by_model[model_key] = draws
            folds_by_model[model_key] = fold_metrics
            feature_counts_by_model[model_key] = feature_count_summary

        combined_predictions = pd.concat(
            [predictions_by_model[key] for key in args.models],
            ignore_index=True,
        )
        combined_predictions.to_csv(
            task_dir / "all_models_oof_predictions.csv",
            index=False,
        )
        combined_predictions.to_excel(
            task_dir / "all_models_oof_predictions.xlsx",
            index=False,
        )
        pilot_level_summary(task, combined_predictions, pd, np).to_csv(
            task_dir / "pilot_summary_by_model.csv", index=False
        )
        combined_folds = pd.concat(
            [folds_by_model[key] for key in args.models],
            ignore_index=True,
        )
        combined_folds.to_csv(task_dir / "all_models_per_fold.csv", index=False)
        combined_modalities = pd.concat(
            [modality_by_model[key] for key in args.models],
            ignore_index=True,
        )
        combined_modalities.to_csv(
            task_dir / "all_models_modality_selection.csv", index=False
        )
        plot_modality_selection(
            combined_modalities,
            task_dir,
            plt,
            np,
            args.figure_formats,
            args.figure_dpi,
        )
        plot_fold_stability(
            task,
            combined_folds,
            task_dir,
            plt,
            args.figure_formats,
            args.figure_dpi,
        )
        combined_stability = pd.concat(
            [
                stability_by_model[key].assign(
                    model=key,
                    model_display_name=MODEL_DISPLAY[key],
                )
                for key in args.models
            ],
            ignore_index=True,
        )
        combined_stability.to_csv(
            task_dir / "all_models_metric_stability.csv",
            index=False,
        )
        combined_draws = pd.concat(
            [draws_by_model[key].assign(model=key) for key in args.models],
            ignore_index=True,
        )
        combined_draws.to_csv(
            task_dir / "all_models_bootstrap_draws.csv",
            index=False,
        )

        comparison_records: list[dict[str, Any]] = []
        for model_key in args.models:
            stability_lookup = stability_by_model[model_key].set_index("metric")
            record: dict[str, Any] = {
                "model": model_key,
                "model_display_name": MODEL_DISPLAY[model_key],
                **feature_counts_by_model[model_key],
            }
            for metric_name, value in metrics_by_model[model_key].items():
                record[metric_name] = float(value)
                record[f"{metric_name}_ci_lower"] = float(
                    stability_lookup.loc[metric_name, "ci_lower"]
                )
                record[f"{metric_name}_ci_upper"] = float(
                    stability_lookup.loc[metric_name, "ci_upper"]
                )
            comparison_records.append(record)
        comparison = pd.DataFrame(comparison_records)
        if task == "classification":
            comparison["rank"] = comparison["roc_auc_hard"].rank(
                method="min",
                ascending=False,
            ).astype(int)
        else:
            comparison["rank"] = comparison["rmse"].rank(
                method="min",
                ascending=True,
            ).astype(int)
        comparison = comparison.sort_values("rank")
        comparison.to_csv(task_dir / "model_comparison.csv", index=False)
        save_json(
            task_dir / "model_comparison.json",
            {
                row["model"]: row
                for row in comparison.to_dict(orient="records")
            },
        )
        paired = paired_differences(
            task=task,
            metrics_by_model=metrics_by_model,
            draws_by_model=draws_by_model,
            confidence_level=args.confidence_level,
            pd=pd,
            np=np,
        )
        paired.to_csv(
            task_dir / "paired_differences_vs_tabfm.csv",
            index=False,
        )
        plot_paired_advantages(
            task,
            paired,
            task_dir,
            plt,
            np,
            args.figure_formats,
            args.figure_dpi,
        )
        plot_task_comparison(
            task,
            comparison,
            task_dir,
            plt,
            np,
            args.figure_formats,
            args.figure_dpi,
        )
        if task == "classification":
            plot_classification_diagnostics(
                predictions_by_model,
                task_dir,
                metric_functions,
                plt,
                np,
                args.figure_formats,
                args.figure_dpi,
            )
            plot_confusion_grid(
                predictions_by_model,
                task_dir,
                plt,
                np,
                args.figure_formats,
                args.figure_dpi,
            )
        else:
            plot_regression_diagnostics(
                predictions_by_model,
                task_dir,
                plt,
                np,
                args.figure_formats,
                args.figure_dpi,
            )
        if "tabfm_ensemble" in predictions_by_model:
            plot_tabfm_ensemble_diagnostics(
                task,
                predictions_by_model["tabfm_ensemble"],
                task_dir,
                pd,
                plt,
                np,
                args.figure_formats,
                args.figure_dpi,
            )
        if shap_modality_frames:
            pd.concat(shap_modality_frames, ignore_index=True).to_csv(
                task_dir / "all_models_shap_modality_importance.csv",
                index=False,
            )
        all_task_summaries[task] = comparison.to_dict(orient="records")

        print(f"\n{TASK_DISPLAY[task]} comparison")
        display_metrics = (
            ["model_display_name", "accuracy", "f1_macro", "roc_auc_hard", "rank"]
            if task == "classification"
            else ["model_display_name", "mae", "rmse", "r2", "rank"]
        )
        print(comparison[display_metrics].to_string(index=False))

        if tabfm_backbone is not None:
            del tabfm_backbone
            gc.collect()
            if torch is not None and device == "cuda":
                torch.cuda.empty_cache()

    save_json(output_dir / "dual_task_summary.json", all_task_summaries)
    print(f"\nCompleted requested tasks. Outputs:\n  {output_dir}")


if __name__ == "__main__":
    main()
