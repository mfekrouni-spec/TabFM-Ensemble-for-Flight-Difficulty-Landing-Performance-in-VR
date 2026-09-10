#!/usr/bin/env python
"""Fast local monitor for persisted TabFM OOF evidence and raw signals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import webbrowser

from flask import Flask, Response, jsonify, render_template_string, request
import pandas as pd

from narrative_explainer import (
    DEFAULT_CLASSIFICATION,
    DEFAULT_REGRESSION,
    DEFAULT_WORKBOOK,
    build_evidence,
    load_shap_rows,
    plain_feature_label,
    verbalize_evidence,
)
from raw_signal_loader import (
    DEFAULT_RAW_ROOT,
    MODALITY_ORDER,
    RawSignalRepository,
)


from workflow_paths import RESULTS
from narrative_store import stored_narrative
from direct_shap import collect_validated
LLM_AUDIT_PATH = RESULTS / "llm_narration_audit.jsonl"
LLM_AUDIT_LOCK = threading.Lock()
WATERFALL_PLOT_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def screenshot_browser() -> Path:
    """Return an installed Chromium browser for origin-safe PNG rendering."""
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    ]
    for executable_name in ("chrome", "chrome.exe", "msedge", "msedge.exe", "google-chrome", "chromium", "chromium-browser"):
        resolved = shutil.which(executable_name)
        if resolved:
            candidates.append(Path(resolved))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Microsoft Edge or Google Chrome is required for automatic PNG export."
    )


def append_llm_audit(evidence: dict, narrative: dict) -> None:
    """Persist local narration evidence without storing any API credential."""
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": evidence.get("task"),
        "source_excel_row": evidence.get("source_excel_row"),
        "subject": evidence.get("subject"),
        "run": evidence.get("run"),
        "level": evidence.get("level"),
        "prompt_version": narrative.get("prompt_version"),
        "evidence_sha256": narrative.get("evidence_sha256"),
        "llm_paraphrase": narrative.get("llm_paraphrase"),
        "llm_audit": narrative.get("llm_audit"),
    }
    with LLM_AUDIT_LOCK:
        with LLM_AUDIT_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str, sort_keys=True) + "\n")


PAGE = (Path(__file__).with_name("pilot_monitor_template.html")).read_text(
    encoding="utf-8"
)


def create_app(
    classification_dir: Path,
    regression_dir: Path,
    workbook: Path,
    raw_root: Path = DEFAULT_RAW_ROOT,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
    task_dirs = {
        "classification": classification_dir.resolve(),
        "regression": regression_dir.resolve(),
    }
    workbook = workbook.resolve()
    raw_repository = RawSignalRepository(raw_root)

    @lru_cache(maxsize=2)
    def catalog(task: str) -> list[dict]:
        path = task_dirs[task] / "all_models_oof_predictions.csv"
        frame = pd.read_csv(path)
        frame = frame.loc[frame["model"].astype(str).eq("tabfm_ensemble")]
        columns = ["source_excel_row", "subject", "run", "level", "fold"]
        return frame[columns].sort_values(
            ["subject", "run", "source_excel_row"]
        ).to_dict(orient="records")

    @lru_cache(maxsize=1000)
    def catalog_row(task: str, source_row: int) -> dict:
        matches = [
            value
            for value in catalog(task)
            if int(value["source_excel_row"]) == int(source_row)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one catalog row for Excel row {source_row}; found {len(matches)}."
            )
        return matches[0]

    def evidence_for(task: str, source_row: int) -> dict:
        return build_evidence(task, source_row, task_dirs[task], workbook)

    def shap_coverage(task):
        _, summary = collect_validated(task_dirs[task])
        return {**summary, "explained_rows": summary["direct_rows"],
                "available_local_rows": summary["direct_rows"], "representative_approximation_rows": 0}

    def shap_cases(task):
        values, summary = collect_validated(task_dirs[task])
        available = set(values.source_excel_row.astype(int)) if not values.empty else set()
        return [{**row, "is_approximation": False, "fidelity_r2": None, "method": "direct_full_ensemble"}
                for row in catalog(task) if int(row["source_excel_row"]) in available]

    def waterfall_png(task: str, source_row: int) -> bytes:
        """Render a standard SHAP local waterfall from persisted values."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import shap

        rows = load_shap_rows(task_dirs[task], source_row)
        if rows.empty:
            raise FileNotFoundError(
                f"No saved local SHAP values for Excel row {source_row}."
            )
        row = catalog_row(task, source_row)
        is_approximation = bool(rows.iloc[0]["_is_approximation"])
        fidelity_raw = rows.iloc[0]["_fidelity_r2"]
        fidelity_r2 = (
            float(fidelity_raw) if pd.notna(fidelity_raw) else None
        )
        scale = 100.0 if task == "classification" else 1.0
        values = rows["shap_value"].astype(float).to_numpy() * scale
        base_value = float(rows.iloc[0]["base_value"]) * scale
        direct_output_column = (
            "probability_hard"
            if task == "classification"
            else "predicted_performance"
        )
        explained_output_column = (
            direct_output_column
            if direct_output_column in rows.columns
            else "fast_tabfm_output"
        )
        explained_output = (
            float(rows.iloc[0][explained_output_column]) * scale
        )
        reconstruction_residual = abs(
            base_value + float(values.sum()) - explained_output
        )
        feature_values = pd.to_numeric(
            rows["feature_value"], errors="coerce"
        ).to_numpy(float)
        feature_names = [
            plain_feature_label(str(feature), str(modality))
            for feature, modality in zip(rows["feature"], rows["modality"])
        ]
        explanation = shap.Explanation(
            values=np.asarray(values, dtype=float),
            base_values=float(base_value),
            data=np.asarray(feature_values, dtype=float),
            feature_names=feature_names,
        )
        output_label = (
            "Hard-class probability (%)"
            if task == "classification"
            else "Predicted landing-performance target"
        )
        method_label = "Direct 32-member ensemble permutation SHAP"
        if is_approximation:
            method_label = "Fast TabFM representative approximation"
            if fidelity_r2 is not None:
                method_label += f" · ensemble OOF output R²={fidelity_r2:.3f}"
        if reconstruction_residual > 1e-6:
            residual_unit = (
                "percentage points"
                if task == "classification"
                else "target units"
            )
            method_label += (
                f" · SHAP reconstruction residual="
                f"{reconstruction_residual:.2f} {residual_unit}"
            )
        with WATERFALL_PLOT_LOCK:
            with plt.rc_context(
                {
                    "font.family": "DejaVu Sans",
                    "font.size": 10,
                    "axes.titlesize": 15,
                    "axes.labelsize": 11,
                }
            ):
                plt.close("all")
                plt.figure(figsize=(13.5, 7.2), facecolor="white")
                shap.plots.waterfall(
                    explanation,
                    max_display=min(12, len(feature_names)),
                    show=False,
                )
                figure = plt.gcf()
                figure.set_size_inches(13.5, 7.2)
                axis = figure.axes[0]
                axis.set_title(
                    f"Pilot {row['subject']} · Run {row['run']} · "
                    f"Level {row['level']} — local SHAP waterfall",
                    loc="left",
                    pad=72,
                    color="#102536",
                    fontweight="bold",
                )
                axis.set_xlabel("")
                axis.text(
                    0.0,
                    1.095,
                    f"{output_label} · {method_label}",
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    color="#526b7d",
                )
                stream = BytesIO()
                figure.savefig(
                    stream,
                    format="png",
                    dpi=170,
                    bbox_inches="tight",
                    facecolor="white",
                )
                plt.close(figure)
        return stream.getvalue()

    @lru_cache(maxsize=128)
    def signal_window(
        subject: int,
        level: int,
        run: int,
        start_seconds: float,
        duration_seconds: float,
        modalities: tuple[str, ...],
    ) -> dict:
        return raw_repository.load_window(
            subject,
            level,
            run,
            start_seconds,
            duration_seconds,
            modalities,
        )

    @app.get("/")
    def index() -> str:
        return render_template_string(PAGE)

    @app.get("/api/catalog")
    def api_catalog():
        task = request.args.get("task", "classification")
        if task not in task_dirs:
            return jsonify({"error": "Unsupported task"}), 400
        return jsonify(catalog(task))

    @app.get("/api/evidence")
    def api_evidence():
        try:
            task = request.args.get("task", "classification")
            if task not in task_dirs:
                raise ValueError("Unsupported task")
            source_row = int(request.args["row"])
            evidence = evidence_for(task, source_row)
            use_llm = request.args.get("llm") == "1"
            narrative = stored_narrative(evidence, use_llm, RESULTS)
            return jsonify({"evidence": evidence, "narrative": narrative})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/shap/cases")
    def api_shap_cases():
        task = request.args.get("task", "classification")
        if task not in task_dirs:
            return jsonify({"error": "Unsupported task"}), 400
        return jsonify(shap_cases(task))

    @app.get("/api/shap/values.csv")
    def api_shap_values():
        try:
            task = request.args.get("task", "classification")
            if task not in task_dirs:
                raise ValueError("Unsupported task")
            row = int(request.args["row"])
            frame = load_shap_rows(task_dirs[task], row)
            if frame.empty:
                raise ValueError("Direct SHAP is not yet complete for this run")
            return Response(frame.to_csv(index=False), mimetype="text/csv",
                headers={"Content-Disposition": f"attachment; filename={task}_row_{row}_direct_shap.csv"})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/shap/waterfall.png")
    def api_shap_waterfall():
        try:
            task = request.args.get("task", "classification")
            if task not in task_dirs:
                raise ValueError("Unsupported task")
            source_row = int(request.args["row"])
            return Response(
                waterfall_png(task, source_row),
                mimetype="image/png",
                headers={"Cache-Control": "no-store"},
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/raw-signals/info")
    def api_raw_signal_info():
        try:
            task = request.args.get("task", "classification")
            row = catalog_row(task, int(request.args["row"]))
            return jsonify(
                raw_repository.session_info(
                    int(row["subject"]), int(row["level"]), int(row["run"])
                )
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/raw-signals/window")
    def api_raw_signal_window():
        try:
            task = request.args.get("task", "classification")
            row = catalog_row(task, int(request.args["row"]))
            modalities = tuple(request.args.getlist("modality"))
            unsupported = [value for value in modalities if value not in MODALITY_ORDER]
            if unsupported:
                raise ValueError(f"Unsupported signal modalities: {unsupported}")
            start = round(float(request.args.get("start", 0)), 3)
            duration = round(float(request.args.get("duration", 30)), 3)
            return jsonify(
                signal_window(
                    int(row["subject"]),
                    int(row["level"]),
                    int(row["run"]),
                    start,
                    duration,
                    modalities,
                )
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/export/png")
    def api_export_png():
        """Render a standalone page snapshot to a 3200x1800 PNG."""
        try:
            task = request.form.get("task", "")
            if task not in task_dirs:
                raise ValueError("Unsupported task")
            source_row = int(request.form["row"])
            row = catalog_row(task, source_row)
            page_name = request.form.get("page", "")
            page_numbers = {"configuration": 1, "results": 2}
            if page_name not in page_numbers:
                raise ValueError("Unsupported export page")
            export_id = request.form.get("export_id", "")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", export_id):
                raise ValueError("Invalid export identifier")
            upload = request.files.get("snapshot")
            if upload is None:
                raise ValueError("HTML page snapshot is missing")
            snapshot = upload.read(30 * 1024 * 1024 + 1)
            if len(snapshot) > 30 * 1024 * 1024:
                raise ValueError("HTML snapshot exceeds the 30 MB export limit")
            if b"<!doctype html>" not in snapshot[:256].lower():
                raise ValueError("Export payload is not a standalone HTML snapshot")
            export_dir = Path(__file__).resolve().parent / "gui_exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"{export_id}_{task}_pilot_{int(row['subject'])}_"
                f"run_{int(row['run'])}_page_{page_numbers[page_name]}_"
                f"{page_name}.png"
            )
            destination = export_dir / filename
            browser = screenshot_browser()
            with tempfile.TemporaryDirectory(
                prefix="pilot-monitor-export-",
                ignore_cleanup_errors=True,
            ) as temporary_name:
                temporary_dir = Path(temporary_name)
                snapshot_path = temporary_dir / "snapshot.html"
                screenshot_path = temporary_dir / "screenshot.png"
                profile_path = temporary_dir / "browser-profile"
                snapshot_path.write_bytes(snapshot)
                command = [
                    str(browser),
                    "--headless=new",
                    "--disable-gpu",
                    "--disable-extensions",
                    "--hide-scrollbars",
                    "--no-first-run",
                    "--no-default-browser-check",
                    f"--user-data-dir={profile_path}",
                    "--window-size=1600,900",
                    "--force-device-scale-factor=2",
                    "--virtual-time-budget=1500",
                    f"--screenshot={screenshot_path}",
                    snapshot_path.as_uri(),
                ]
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=45,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if screenshot_path.is_file() and screenshot_path.stat().st_size >= 24:
                        break
                    time.sleep(0.1)
                if not screenshot_path.is_file():
                    detail = (completed.stderr or completed.stdout).strip()
                    raise RuntimeError(
                        "The local browser could not render the PNG"
                        + (f": {detail[:500]}" if detail else ".")
                    )
                image = screenshot_path.read_bytes()
                if len(image) < 24 or image[:8] != b"\x89PNG\r\n\x1a\n":
                    raise RuntimeError("The local browser returned an invalid PNG")
                width, height = struct.unpack(">II", image[16:24])
                if (width, height) != (3200, 1800):
                    raise RuntimeError(
                        f"Expected a 3200x1800 PNG; received {width}x{height}"
                    )
                screenshot_path.replace(destination)
            return jsonify(
                {
                    "browser": browser.name,
                    "bytes": len(image),
                    "directory": str(export_dir),
                    "filename": filename,
                    "height": height,
                    "path": str(destination),
                    "width": width,
                }
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/runtime-status")
    def api_runtime_status():
        return jsonify(
            {
                "llm_ready": bool(os.environ.get("OPENROUTER_API_KEY")),
                "llm_provider": "OpenRouter",
                "llm_model": os.environ.get(
                    "OPENROUTER_MODEL", "openrouter/free"
                ),
                "raw_signal_runs": len(raw_repository.runs),
                "shap_coverage": {
                    task: shap_coverage(task) for task in task_dirs
                },
                "shap_mode": "verified_direct_full_ensemble_only",
            }
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-dir", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--regression-dir", type=Path, default=DEFAULT_REGRESSION)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--open-browser", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [args.workbook, *[
        directory / "all_models_oof_predictions.csv"
        for directory in (args.classification_dir, args.regression_dir)
    ]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing GUI inputs: " + ", ".join(missing)
            + ". Configure the workbook, run run_benchmarks.py for both tasks, "
            "then run_all_shap.py. See README.md.")
    app = create_app(
        args.classification_dir,
        args.regression_dir,
        args.workbook,
        args.raw_root,
    )
    url = f"http://{args.host}:{args.port}"
    if args.open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Pilot monitor: {url}")
    print("Predictions, SHAP, and raw-signal windows are read from saved local files.")
    print("Press Ctrl+C to stop.")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
