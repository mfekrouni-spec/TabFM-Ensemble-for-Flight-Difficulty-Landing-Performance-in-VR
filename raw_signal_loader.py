#!/usr/bin/env python
"""Indexed, windowed access to the raw multimodal VR landing recordings."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


from workflow_paths import configured_path
DEFAULT_RAW_ROOT = configured_path("raw_root")
RUN_PATTERN = re.compile(r"level-(\d{2})B_run-(\d{3})$")
STREAM_PATTERN = re.compile(r"_stream-(.+?)_feat-")

MODALITY_ORDER = (
    "Respiration",
    "PPG",
    "Head Movement",
    "Forearm Accel.",
    "EMG",
    "ECG/HRV",
    "Eye Movement",
    "Torso Accel.",
    "EDA",
)

# Candidate streams are ordered by preference. The UI exposes physiological or
# movement waveforms; derived workbook features remain a separate display.
MODALITY_SPECS: dict[str, tuple[dict[str, Any], ...]] = {
    "Respiration": (
        {
            "stream": "lslshimmerresp",
            "series": (("respiration_trace_mV", "Respiration", "mV"),),
        },
        {
            "stream": "lslrespitrace",
            "series": (("respiration_trace_v", "Respiration", "V"),),
        },
    ),
    "PPG": (
        {
            "stream": "lslshimmereda",
            "series": (("ppg_finger_mV", "Finger PPG", "mV"),),
        },
    ),
    "Head Movement": (
        {
            "stream": "lslxp11xpcplt",
            "series": (
                ("pilot_head_roll_deg", "Roll", "deg"),
                ("pilot_head_pitch_deg", "Pitch", "deg"),
                ("pilot_head_yaw_deg", "Yaw", "deg"),
            ),
        },
    ),
    "Forearm Accel.": (
        {
            "stream": "lslshimmeremg",
            "series": (
                ("accelerometry_forearm_r_x_mps2", "X", "m/s²"),
                ("accelerometry_forearm_r_y_mps2", "Y", "m/s²"),
                ("accelerometry_forearm_r_z_mps2", "Z", "m/s²"),
            ),
        },
    ),
    "EMG": (
        {
            "stream": "lslshimmeremg",
            "series": (
                ("emg_wrist_flexor_mV", "Wrist flexor", "mV"),
                ("emg_wrist_extensor_mV", "Wrist extensor", "mV"),
            ),
        },
    ),
    "ECG/HRV": (
        {
            "stream": "lslshimmerecg",
            "series": (
                ("ecg_projection_ll_ra_mV", "LL–RA", "mV"),
                ("ecg_projection_la_ra_mV", "LA–RA", "mV"),
            ),
        },
    ),
    "Eye Movement": (
        {
            "stream": "lslhtcviveeye",
            "series": (
                ("pupil_diameter_l_mm", "Left pupil", "mm"),
                ("pupil_diameter_r_mm", "Right pupil", "mm"),
            ),
        },
    ),
    "Torso Accel.": (
        {
            "stream": "lslshimmertorsoacc",
            "series": (
                ("accelerometry_torso_x_mps2", "X", "m/s²"),
                ("accelerometry_torso_y_mps2", "Y", "m/s²"),
                ("accelerometry_torso_z_mps2", "Z", "m/s²"),
            ),
        },
    ),
    "EDA": (
        {
            "stream": "lslshimmereda",
            "series": (("eda_hand_l_kOhms", "Hand EDA", "kΩ"),),
        },
    ),
}


@dataclass(frozen=True)
class StreamInfo:
    name: str
    data_path: Path
    columns: tuple[str, ...]
    sample_count: int
    sampling_hz: float | None
    first_time_dn: float | None


@dataclass(frozen=True)
class RunInfo:
    subject: int
    level: int
    run: int
    path: Path
    streams: dict[str, StreamInfo]


def _first_data_timestamp(path: Path) -> float | None:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            handle.readline()
            line = handle.readline()
        return float(line.split(",", 1)[0]) if line else None
    except (OSError, ValueError):
        return None


def _header_metadata(path: Path) -> tuple[int, float | None]:
    if not path.is_file():
        return 0, None
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            row = next(csv.DictReader(handle))
        sample_count = int(float(row.get("sampleCount") or 0))
        raw_hz = row.get("Fs_Hz_effective")
        sampling_hz = float(raw_hz) if raw_hz not in {None, "", "NaN"} else None
        if sampling_hz is not None and (
            not math.isfinite(sampling_hz) or sampling_hz <= 0
        ):
            sampling_hz = None
        return sample_count, sampling_hz
    except (OSError, StopIteration, ValueError):
        return 0, None


def _downsample_extrema(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
) -> tuple[list[float], list[float]]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) <= max_points:
        return x.astype(float).tolist(), y.astype(float).tolist()
    bucket_count = max(1, max_points // 2)
    edges = np.linspace(0, len(x), bucket_count + 1, dtype=int)
    selected: list[int] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        local = y[left:right]
        low = left + int(np.argmin(local))
        high = left + int(np.argmax(local))
        selected.extend(sorted({low, high}))
    indices = np.asarray(sorted(set(selected)), dtype=int)
    return x[indices].astype(float).tolist(), y[indices].astype(float).tolist()


class RawSignalRepository:
    """Scan lightweight metadata once and load only requested signal windows."""

    def __init__(self, root: Path = DEFAULT_RAW_ROOT) -> None:
        self.root = root.resolve()
        self.runs: dict[tuple[int, int, int], RunInfo] = {}
        self._scan()

    def _scan(self) -> None:
        for subject_dir in sorted(self.root.glob("sub-cp*")):
            try:
                subject = int(subject_dir.name.replace("sub-cp", ""))
            except ValueError:
                continue
            for run_dir in sorted(subject_dir.glob("ses-*/*")):
                match = RUN_PATTERN.match(run_dir.name)
                if not match:
                    continue
                level, run = (int(value) for value in match.groups())
                streams: dict[str, StreamInfo] = {}
                for data_path in run_dir.glob("*_dat.csv"):
                    stream_match = STREAM_PATTERN.search(data_path.name)
                    if not stream_match:
                        continue
                    stream = stream_match.group(1)
                    with data_path.open(
                        "r", encoding="utf-8-sig", errors="replace"
                    ) as handle:
                        columns = tuple(handle.readline().strip().split(","))
                    header_path = data_path.with_name(
                        data_path.name.replace("_dat.csv", "_hea.csv")
                    )
                    sample_count, sampling_hz = _header_metadata(header_path)
                    streams[stream] = StreamInfo(
                        name=stream,
                        data_path=data_path,
                        columns=columns,
                        sample_count=sample_count,
                        sampling_hz=sampling_hz,
                        first_time_dn=_first_data_timestamp(data_path),
                    )
                self.runs[(subject, level, run)] = RunInfo(
                    subject=subject,
                    level=level,
                    run=run,
                    path=run_dir,
                    streams=streams,
                )

    def _run(self, subject: int, level: int, run: int) -> RunInfo:
        key = (int(subject), int(level), int(run))
        if key not in self.runs:
            raise KeyError(
                f"No raw-signal folder for pilot {subject}, level {level}, run {run}."
            )
        return self.runs[key]

    @staticmethod
    def _candidate(
        run_info: RunInfo,
        modality: str,
    ) -> tuple[StreamInfo, dict[str, Any]] | None:
        for specification in MODALITY_SPECS[modality]:
            stream = run_info.streams.get(str(specification["stream"]))
            columns = {value[0] for value in specification["series"]}
            if (
                stream is not None
                and stream.sample_count > 0
                and columns.issubset(stream.columns)
            ):
                return stream, specification
        return None

    def session_info(self, subject: int, level: int, run: int) -> dict[str, Any]:
        run_info = self._run(subject, level, run)
        available = [
            modality
            for modality in MODALITY_ORDER
            if self._candidate(run_info, modality) is not None
        ]
        durations = [
            stream.sample_count / stream.sampling_hz
            for stream in run_info.streams.values()
            if stream.sample_count > 0 and stream.sampling_hz
        ]
        return {
            "subject": subject,
            "level": level,
            "run": run,
            "available_modalities": available,
            "unavailable_modalities": [
                value for value in MODALITY_ORDER if value not in available
            ],
            "estimated_duration_seconds": (
                round(max(durations), 1) if durations else None
            ),
            "source_path": str(run_info.path),
        }

    @staticmethod
    def _read_window(
        stream: StreamInfo,
        columns: set[str],
        anchor_dn: float,
        start_seconds: float,
        duration_seconds: float,
    ) -> pd.DataFrame:
        target_start = anchor_dn + start_seconds / 86400.0
        target_end = target_start + duration_seconds / 86400.0
        use_columns = ["time_dn", *sorted(columns)]
        row_start = 0
        row_count: int | None = None
        if stream.sampling_hz and stream.first_time_dn is not None:
            offset_seconds = max(
                0.0, (target_start - stream.first_time_dn) * 86400.0
            )
            margin = max(1, int(math.ceil(stream.sampling_hz)))
            row_start = max(0, int(offset_seconds * stream.sampling_hz) - margin)
            row_count = int(
                math.ceil((duration_seconds + 3.0) * stream.sampling_hz)
            )
        read_args: dict[str, Any] = {
            "usecols": use_columns,
            "low_memory": False,
        }
        if row_start:
            read_args.update(
                {
                    "header": None,
                    "names": list(stream.columns),
                    "skiprows": row_start + 1,
                }
            )
        if row_count is not None:
            read_args["nrows"] = row_count
        frame = pd.read_csv(stream.data_path, **read_args)
        frame["time_dn"] = pd.to_numeric(frame["time_dn"], errors="coerce")
        frame = frame.loc[
            frame["time_dn"].between(target_start, target_end, inclusive="both")
        ].copy()
        frame["time_seconds"] = (frame["time_dn"] - anchor_dn) * 86400.0
        return frame

    def load_window(
        self,
        subject: int,
        level: int,
        run: int,
        start_seconds: float,
        duration_seconds: float,
        modalities: tuple[str, ...],
        max_points_per_series: int = 1000,
    ) -> dict[str, Any]:
        if start_seconds < 0:
            raise ValueError("Signal start must be zero or greater.")
        if not 1 <= duration_seconds <= 300:
            raise ValueError("Signal duration must be between 1 and 300 seconds.")
        requested = tuple(
            value for value in MODALITY_ORDER if value in set(modalities)
        )
        if not requested:
            raise ValueError("Select at least one available signal modality.")
        run_info = self._run(subject, level, run)
        chosen: dict[str, tuple[StreamInfo, dict[str, Any]]] = {}
        unavailable: list[str] = []
        for modality in requested:
            candidate = self._candidate(run_info, modality)
            if candidate is None:
                unavailable.append(modality)
            else:
                chosen[modality] = candidate
        timestamps = [
            value.first_time_dn
            for value in run_info.streams.values()
            if value.sample_count > 0 and value.first_time_dn is not None
        ]
        if not timestamps:
            raise RuntimeError("No timestamped raw streams are available for this run.")
        anchor_dn = min(timestamps)

        stream_requests: dict[str, dict[str, Any]] = {}
        for modality, (stream, specification) in chosen.items():
            request_entry = stream_requests.setdefault(
                stream.name,
                {"stream": stream, "columns": set(), "modalities": []},
            )
            request_entry["columns"].update(
                value[0] for value in specification["series"]
            )
            request_entry["modalities"].append((modality, specification))

        plots: list[dict[str, Any]] = []
        for request_entry in stream_requests.values():
            stream = request_entry["stream"]
            frame = self._read_window(
                stream,
                request_entry["columns"],
                anchor_dn,
                start_seconds,
                duration_seconds,
            )
            x = frame["time_seconds"].to_numpy(dtype=float)
            for modality, specification in request_entry["modalities"]:
                series_payload = []
                for column, label, unit in specification["series"]:
                    y = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                        dtype=float
                    ).copy()
                    if column.startswith("pupil_diameter"):
                        y[y <= 0] = np.nan
                    x_values, y_values = _downsample_extrema(
                        x, y, max_points_per_series
                    )
                    if y_values:
                        series_payload.append(
                            {
                                "name": label,
                                "column": column,
                                "unit": unit,
                                "x": x_values,
                                "y": y_values,
                            }
                        )
                plots.append(
                    {
                        "modality": modality,
                        "stream": stream.name,
                        "series": series_payload,
                        "samples_in_window": int(len(frame)),
                    }
                )
        plots.sort(key=lambda value: MODALITY_ORDER.index(value["modality"]))
        return {
            "subject": subject,
            "level": level,
            "run": run,
            "start_seconds": float(start_seconds),
            "duration_seconds": float(duration_seconds),
            "plots": plots,
            "unavailable_modalities": unavailable,
            "source_path": str(run_info.path),
        }
