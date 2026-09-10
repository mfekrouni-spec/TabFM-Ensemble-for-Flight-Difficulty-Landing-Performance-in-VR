"""Portable configuration shared by the benchmark, direct SHAP and GUI."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "workflow_config.json").read_text(encoding="utf-8-sig"))
override = os.environ.get("TABFM_CONFIG")
CONFIG_OVERRIDE = Path(override).expanduser().resolve() if override else ROOT / "workflow_config.local.json"
if override and not CONFIG_OVERRIDE.is_file():
    raise FileNotFoundError(f"TABFM_CONFIG does not exist: {CONFIG_OVERRIDE}")
if CONFIG_OVERRIDE.is_file():
    CONFIG.update(json.loads(CONFIG_OVERRIDE.read_text(encoding="utf-8-sig")))

def configured_path(key):
    path = Path(CONFIG[key]).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()

RESULTS = configured_path("results_root")
WORKBOOK = configured_path("workbook")

def task_dir(task):
    return RESULTS / task

def resolve_runtime(torch, device=None, dtype_name=None):
    device = CONFIG["device"] if device is None else device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is not available in this PyTorch environment")
    dtype_name = CONFIG["dtype"] if dtype_name is None else dtype_name
    if dtype_name == "auto":
        dtype_name = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    if dtype_name not in ("float32", "bfloat16"):
        raise ValueError("dtype must be auto, float32, or bfloat16")
    if device == "cuda" and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16; configure float32")
    return device, getattr(torch, dtype_name)
