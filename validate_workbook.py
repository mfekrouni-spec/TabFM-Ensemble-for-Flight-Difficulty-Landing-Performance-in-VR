"""Validate the feature-table contract shared by the benchmark, SHAP and GUI."""
from pathlib import Path

import numpy as np
import pandas as pd

METADATA = ["Subject", "level", "run", "flight_hours", "performance"]

def validate_workbook(path):
    frame = pd.read_excel(Path(path), sheet_name="data")
    if frame.empty or len(frame.columns) <= 5:
        raise ValueError("The data sheet needs rows, five metadata columns and numeric predictors")
    if list(frame.columns[:5]) != METADATA:
        raise ValueError("The first five columns must be ordered: " + ", ".join(METADATA))
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported")
    for name in METADATA:
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite numeric values")
        if name in ("Subject", "level", "run") and not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{name} must contain integer identifiers/levels")
    if not frame.level.isin([1, 2, 3, 4]).all():
        raise ValueError("level must be 1, 2, 3, or 4")
    if frame.duplicated(["Subject", "level", "run"]).any():
        raise ValueError("Each Subject/level/run combination must identify one row")
    for name in frame.columns[5:]:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f"Predictor {name} is not numeric")
        if not np.isfinite(frame[name].to_numpy(float)).any():
            raise ValueError(f"Predictor {name} has no finite observations")
    return {"rows": len(frame), "predictors": len(frame.columns) - 5,
            "subjects": int(frame.Subject.nunique()), "sheet": "data"}

if __name__ == "__main__":
    import argparse
    import json
    from workflow_paths import WORKBOOK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", nargs="?", type=Path, default=WORKBOOK)
    print(json.dumps(validate_workbook(parser.parse_args().workbook), indent=2))
