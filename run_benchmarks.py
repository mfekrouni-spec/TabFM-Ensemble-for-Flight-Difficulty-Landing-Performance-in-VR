"""Create the benchmark artifacts needed by run_all_shap.py from a feature workbook."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from workflow_paths import ROOT, CONFIG, WORKBOOK, configured_path
from direct_shap import atomic_json, digest_file

PACKAGES = ("numpy", "pandas", "shap", "torch", "tabfm", "scikit-learn", "joblib")

def guard_invocation(task, command):
    """Do not relabel cached results as outputs of changed code/environments."""
    directory = configured_path("source_root") / f"tabfm_{task}_optimized_results"
    models_start = command.index("--models") + 1
    models_end = command.index("--no-explain")
    models = command[models_start:models_end]
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "openpyxl"]
    if "xgboost" in models:
        packages.append("xgboost")
    checkpoints, runtime = {}, {}
    if "tabfm_ensemble" in models:
        import torch
        from workflow_paths import resolve_runtime
        packages += ["torch", "tabfm", "safetensors"]
        device, dtype = resolve_runtime(torch)
        runtime = {"device": device, "dtype": str(dtype)}
        checkpoint = configured_path("checkpoint_root") / task
        checkpoints = {name: digest_file(checkpoint / name) for name in ("config.json", "model.safetensors")}
    current = {"schema": 1, "command": command, "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "source": {name: digest_file(ROOT / name) for name in
                   ("tabfm_dual_task_optimized.py", "run_benchmarks.py", "workflow_paths.py")},
        "workbook_sha256": digest_file(WORKBOOK), "checkpoints": checkpoints, "runtime": runtime}
    path = directory / "benchmark_environment.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != current:
            raise ValueError("Benchmark code, data, arguments or environment changed. Choose a new source_root and results_root.")
    elif (directory / "run_manifest.json").exists():
        raise ValueError("Unindexed prior benchmark outputs: use a new source_root for a new benchmark. Existing manuscript outputs may be read directly by run_all_shap.py.")
    else:
        atomic_json(path, current)

def command_for(task, settings, models=None, validate_only=False):
    command = [sys.executable, "-u", str(ROOT / "tabfm_dual_task_optimized.py"),
        "--input", str(WORKBOOK), "--checkpoint-path", str(configured_path("checkpoint_root")),
        "--output-dir", str(configured_path("source_root") / f"tabfm_{task}_optimized_results"),
        "--tasks", task, "--models", *(models or settings["models"]), "--no-explain",
        "--device", CONFIG["device"], "--dtype", CONFIG["dtype"]]
    for key, value in settings.items():
        if key in ("tasks", "models"):
            continue
        command += ["--" + key.replace("_", "-")]
        command += list(map(str, value)) if isinstance(value, list) else [str(value)]
    if validate_only:
        command.append("--validate-only")
    return command

def record_states(task):
    directory = configured_path("source_root") / f"tabfm_{task}_optimized_results"
    states = sorted((directory / task / "models/tabfm_ensemble/folds").glob("*_tabfm_fold_state.joblib"))
    if not states:
        return
    checkpoint = configured_path("checkpoint_root") / task
    index = dict(schema=1, run_manifest_sha256=digest_file(directory / "run_manifest.json"),
        benchmark_source_sha256=digest_file(ROOT / "tabfm_dual_task_optimized.py"),
        packages={name: importlib.metadata.version(name) for name in PACKAGES},
        checkpoints={name: digest_file(checkpoint / name) for name in ("config.json", "model.safetensors")},
        states={path.name: digest_file(path) for path in states})
    atomic_json(directory / "state_index.json", index)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression", "both"), default="both")
    parser.add_argument("--models", nargs="+", choices=("tabfm_ensemble", "xgboost", "random_forest", "knn"))
    parser.add_argument("--settings", default=str(ROOT / "config/manuscript.json"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without data access or model fitting")
    args = parser.parse_args()
    settings = json.loads(open(args.settings, encoding="utf-8-sig").read())
    if int(CONFIG["seed"]) != int(settings["seed"]):
        raise ValueError("The workflow and benchmark settings must use the same seed")
    if not args.dry_run and not WORKBOOK.is_file():
        raise FileNotFoundError(f"Feature workbook missing: {WORKBOOK}. See docs/DATA.md.")
    if not args.dry_run:
        from validate_workbook import validate_workbook
        print("Feature workbook contract: " + json.dumps(validate_workbook(WORKBOOK)), flush=True)
    for task in ("classification", "regression") if args.task == "both" else (args.task,):
        command = command_for(task, settings, args.models, args.validate_only)
        print(subprocess.list2cmdline(command), flush=True)
        if args.dry_run:
            continue
        if not args.validate_only:
            guard_invocation(task, command)
        subprocess.run(command, check=True, cwd=ROOT)
        if not args.validate_only:
            record_states(task)

if __name__ == "__main__":
    main()
