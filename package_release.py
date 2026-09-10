"""Create checksums and a source-only ZIP from an explicit publication allowlist."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = """
.env.example
.gitattributes
.gitignore
0_RUN_BENCHMARKS.cmd
1_COMPUTE_ALL_SHAP.cmd
2_OPEN_GUI.cmd
3_OPTIONAL_PRECOMPUTE_LLM.cmd
AUTHORS.md
CHANGELOG.md
CITATION.cff
LICENSE
NOTICE
README.md
SOURCE_PROVENANCE.json
PRECOMPUTE_ALL_SHAP.ps1
PRECOMPUTE_LLM.ps1
RUN_BENCHMARKS.ps1
RUN_PILOT_MONITOR.ps1
VALIDATE_INPUTS.ps1
direct_shap.py
download_checkpoints.py
narrative_explainer.py
narrative_store.py
package_release.py
pilot_monitor_app.py
pilot_monitor_template.html
precompute_narratives.py
raw_signal_loader.py
requirements-dev.txt
requirements.txt
run_all_shap.py
run_benchmarks.py
summarize_llm_audit.py
tabfm_dual_task_optimized.py
validate_workbook.py
workflow_common.ps1
workflow_config.json
workflow_paths.py
config/manuscript.json
data/feature_schema.csv
docs/DATA.md
docs/REPRODUCIBILITY.md
docs/VALIDATION.md
docs/requirements-environment-snapshot.txt
tests/test_direct_workflow.py
tests/test_repository.py
""".split()

def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def source_files():
    expected = set(SOURCE_FILES)
    for folder in ("config", "docs", "tests"):
        actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts}
        unexpected = actual - expected
        if unexpected:
            raise ValueError(f"Review unexpected source-directory files before packaging: {sorted(unexpected)}")
    for name in sorted(expected):
        path = ROOT / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Missing or invalid publication source: {name}")
    return sorted(expected)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT.parent / (ROOT.name + ".zip"))
    args = parser.parse_args()
    names = source_files()
    manifest = {"algorithm": "SHA-256", "generated_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Publication source files only; this manifest excludes itself.",
                "files": {name: sha256(ROOT / name) for name in names}}
    manifest_path = ROOT / "PACKAGE_SHA256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Checksummed {len(names)} source files.")
    if args.manifest_only:
        return
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("Package output must have a .zip extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in names + ["PACKAGE_SHA256.json"]:
            bundle.write(ROOT / name, arcname=name)
    with zipfile.ZipFile(output) as bundle:
        if bundle.testzip() is not None or set(bundle.namelist()) != set(names + ["PACKAGE_SHA256.json"]):
            raise RuntimeError("Package verification failed")
        for name in names:
            if hashlib.sha256(bundle.read(name)).hexdigest() != manifest["files"][name]:
                raise RuntimeError(f"Package hash mismatch: {name}")
    print(f"Verified {len(names) + 1} ZIP members: {output}")
    print(f"ZIP SHA-256: {sha256(output)}")

if __name__ == "__main__":
    main()
