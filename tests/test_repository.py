"""Portable workflow and audit tests; no network, checkpoints or participant data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from run_benchmarks import command_for
import run_benchmarks
from summarize_llm_audit import summarize
from validate_workbook import validate_workbook
from workflow_paths import resolve_runtime

class RepositoryTests(unittest.TestCase):
    def test_resume_guard_rejects_changed_source_or_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "workbook.xlsx"
            workbook.write_bytes(b"test identity fixture")
            with patch.object(run_benchmarks, "WORKBOOK", workbook), \
                 patch.object(run_benchmarks, "configured_path", return_value=root):
                command = [sys.executable, "benchmark", "--models", "knn", "--no-explain"]
                run_benchmarks.guard_invocation("classification", command)
                run_benchmarks.guard_invocation("classification", command)
                with self.assertRaisesRegex(ValueError, "changed"):
                    run_benchmarks.guard_invocation("classification", command + ["--seed", "13"])
    def test_manuscript_command_preserves_cv_and_members(self):
        settings = json.loads((ROOT / "config/manuscript.json").read_text())
        for task in ("classification", "regression"):
            command = command_for(task, settings, validate_only=True)
            for flag, expected in [("--folds", "5"), ("--inner-folds", "3"),
                                   ("--tabfm-estimators", "32"), ("--tasks", task)]:
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertIn("--no-explain", command)
            self.assertIn("--validate-only", command)

    def test_config_override_paths_are_root_relative_from_other_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            override = Path(directory) / "config.json"
            override.write_text(json.dumps({"workbook": "data/custom.xlsx", "device": "cpu"}))
            env = dict(os.environ, TABFM_CONFIG=str(override), PYTHONPATH=str(ROOT))
            result = subprocess.run([sys.executable, "-c",
                "import json,workflow_paths as p; print(json.dumps([str(p.WORKBOOK),p.CONFIG['device']]))"],
                cwd=directory, env=env, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), [str(ROOT / "data/custom.xlsx"), "cpu"])

    def test_workbook_contract_accepts_fixture_and_rejects_bad_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.xlsx"
            frame = pd.DataFrame({"Subject": [1, 1, 2, 2], "level": [1., 2., 3., 4.],
                "run": [1, 2, 1, 2], "flight_hours": [10., 10., 20., 20.],
                "performance": [1., 2., 3., 4.], "HRV_SDNN": [1., 2., 3., 4.]})
            original = frame.copy()
            frame.to_excel(path, sheet_name="data", index=False)
            self.assertEqual(validate_workbook(path)["rows"], 4)
            frame.loc[1, ["Subject", "level", "run"]] = frame.loc[0, ["Subject", "level", "run"]]
            frame.to_excel(path, sheet_name="data", index=False)
            with self.assertRaisesRegex(ValueError, "one row"):
                validate_workbook(path)
            frame = original
            frame.loc[0, "level"] = 1.5
            frame.to_excel(path, sheet_name="data", index=False)
            with self.assertRaisesRegex(ValueError, "integer"):
                validate_workbook(path)

    def test_runtime_dtype_respects_hardware(self):
        cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False)
        torch = SimpleNamespace(cuda=cuda, float32="float32", bfloat16="bfloat16")
        self.assertEqual(resolve_runtime(torch, "auto", "auto"), ("cuda", "float32"))
        with self.assertRaisesRegex(RuntimeError, "bfloat16"):
            resolve_runtime(torch, "cuda", "bfloat16")
        cuda.is_available = lambda: False
        self.assertEqual(resolve_runtime(torch, "auto", "auto"), ("cpu", "float32"))
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            resolve_runtime(torch, "cuda", "float32")

    def test_audit_denominators_and_empty_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            self.assertIsNone(summarize(path)["acceptance_fraction"])
            records = [dict(task="classification", source_excel_row=2, llm_audit={
                "status": "accepted", "checks": {"length": True, "numbers_in_evidence": True}}),
                dict(task="classification", source_excel_row=3, llm_audit={
                "status": "gate_rejected", "checks": {"length": True, "numbers_in_evidence": False}}),
                dict(task="classification", source_excel_row=3, llm_audit={"status": "request_failed"})]
            path.write_text("\n".join(map(json.dumps, records)))
            result = summarize(path)
            self.assertEqual(result["attempts"], 3)
            self.assertEqual(result["unique_task_run_records"], 2)
            self.assertEqual(result["fallback_attempts"], 2)
            self.assertEqual(result["checks"]["numbers_in_evidence"],
                             {"evaluated": 2, "passed": 1, "failed": 1, "pass_fraction": .5})

    def test_actual_llm_gate_rejects_invented_number(self):
        import narrative_explainer as module
        response = SimpleNamespace(model="mock-model", choices=[SimpleNamespace(
            message=SimpleNamespace(content="The model predicts hard. SHAP effect 999."))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
        evidence = {"task": "classification", "shap_available": True, "prediction": {"predicted_class": "hard"}}
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-not-a-credential"}), \
             patch("openai.OpenAI", return_value=client), \
             patch.object(module, "_detailed_llm_payload", return_value={"probability": 0.5}):
            text, audit = module._llm_paraphrase(evidence, {})
            self.assertIsNone(text)
            self.assertFalse(audit["checks"]["numbers_in_evidence"])
            response.choices[0].message.content = "The model predicts hard. SHAP effect 0.5."
            text, audit = module._llm_paraphrase(evidence, {})
            self.assertTrue(audit["used"])

if __name__ == "__main__":
    unittest.main()
