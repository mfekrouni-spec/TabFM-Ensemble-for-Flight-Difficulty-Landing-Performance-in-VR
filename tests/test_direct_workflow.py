"""CPU tests with small known models; no TabFM fitting or network requests."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import shap

from direct_shap import explain_fold_with_shap, atomic_csv, collect_validated, meta_path, write_summary
from narrative_explainer import load_shap_rows
from narrative_store import stored_narrative

class KnownModel:
    n_estimators = 32
    classes_ = np.array(["hard", "easy"])  # Verify we do not assume column 1.

    def predict_proba(self, values):
        x = np.asarray(values)
        probability = 0.2 + 0.07 * x[:, 0] + 0.03 * x[:, 1]
        return np.column_stack((probability, 1 - probability))

    def predict(self, values):
        x = np.asarray(values)
        return 10 + 3 * x[:, 0] - 2 * x[:, 1]

class DirectWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = KnownModel()
        self.train = pd.DataFrame([[0., 0.], [1., 1.], [2., 1.], [3., 2.]], columns=["HRV_SDNN", "scl_mean"])
        self.test = pd.DataFrame([[1.5, 1.2], [2.3, 0.4]], columns=self.train.columns)
        self.args = SimpleNamespace(seed=42, shap_method="permutation", shap_background_size=4,
            shap_explain_rows=0, shap_source_rows=None, shap_max_evals=0, shap_permutations=3,
            shap_batch_size=5, shap_context_fingerprint="small_known_model_test")

    def tearDown(self):
        self.temp.cleanup()

    def setup_task(self, task="classification"):
        directory = self.root / task
        folds = directory / "models/tabfm_ensemble/folds"
        folds.mkdir(parents=True)
        predictions = pd.DataFrame(dict(model="tabfm_ensemble", source_excel_row=[2, 3], fold=1,
            subject=[1, 1], run=[1, 2], level=[1, 4], flight_hours=[10., 10.]))
        if task == "classification":
            probabilities = self.model.predict_proba(self.test)[:, 0]
            predictions["actual_class"] = ["easy", "hard"]
            predictions["predicted_class"] = np.where(probabilities >= .5, "hard", "easy")
            predictions["probability_hard"] = probabilities
            predictions["probability_easy"] = 1 - probabilities
            predictions["correct"] = predictions.actual_class.eq(predictions.predicted_class)
        else:
            predictions["actual_performance"] = [12., 14.]
            predictions["predicted_performance"] = self.model.predict(self.test)
            predictions["residual"] = predictions.actual_performance - predictions.predicted_performance
            predictions["absolute_error"] = predictions.residual.abs()
        for path in (folds / "fold_01_predictions.csv", directory / "models/tabfm_ensemble/oof_predictions.csv",
                     directory / "all_models_oof_predictions.csv"):
            atomic_csv(path, predictions)
        return directory, folds / "fold_01_shap_values.csv"

    def explain(self, task, path):
        explain_fold_with_shap(task, "tabfm_ensemble", 1, self.model, self.train, self.test,
            np.array([0, 1]), np.array([2, 3]), np.array([1, 1]), path, self.args, shap)

    def test_full_classification_and_known_contributions(self):
        directory, path = self.setup_task()
        self.explain("classification", path)
        frame, summary = collect_validated(directory)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["direct_rows"], 2)
        first = frame[frame.source_excel_row.eq(2)].set_index("feature")
        expected = (self.test.iloc[0] - self.train.mean()) * np.array([.07, .03])
        np.testing.assert_allclose(first.loc[expected.index, "shap_value"], expected, atol=1e-12)

    def test_regression_and_resume_are_identical(self):
        directory, path = self.setup_task("regression")
        self.explain("regression", path)
        original = pd.read_csv(path)
        atomic_csv(path, original[original.source_excel_row.eq(2)])
        self.explain("regression", path)
        pd.testing.assert_frame_equal(original, pd.read_csv(path))
        self.assertTrue(write_summary(directory, plots=True)["complete"])
        self.assertTrue((directory / "models/tabfm_ensemble/figure_shap_beeswarm.png").is_file())

    def test_changed_settings_and_partial_records_rejected(self):
        directory, path = self.setup_task()
        self.explain("classification", path)
        self.args.shap_permutations = 4
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.explain("classification", path)
        self.args.shap_permutations = 3
        frame = pd.read_csv(path)
        atomic_csv(path, frame.iloc[1:])
        _, summary = collect_validated(directory)
        self.assertFalse(summary["complete"])
        self.assertTrue(summary["errors"])
        with self.assertRaises(ValueError):
            load_shap_rows(directory, 2)

    def test_approximation_is_never_loaded(self):
        directory, _ = self.setup_task()
        fast = directory / "models/tabfm_ensemble/fast_tabfm_shap/shap_values.csv"
        atomic_csv(fast, pd.DataFrame(dict(source_excel_row=[2], feature=["HRV_SDNN"], shap_value=[.1])))
        self.assertTrue(load_shap_rows(directory, 2).empty)

    def test_output_mismatch_blocks_shap(self):
        directory, path = self.setup_task()
        prediction = path.with_name("fold_01_predictions.csv")
        frame = pd.read_csv(prediction)
        frame["probability_hard"] += .1
        atomic_csv(prediction, frame)
        with self.assertRaisesRegex(ValueError, "OOF prediction"):
            self.explain("classification", path)
        self.assertFalse(path.exists())

    def test_incomplete_coverage_not_counted_as_complete(self):
        directory, path = self.setup_task()
        self.args.shap_source_rows = [2]
        self.explain("classification", path)
        _, summary = collect_validated(directory)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["missing_source_rows"], [3])

    def test_reduced_ensemble_rejected(self):
        _, path = self.setup_task()
        self.model.n_estimators = 1
        with self.assertRaisesRegex(ValueError, "32-member"):
            self.explain("classification", path)

    def test_gui_refresh_plots_and_cached_llm(self):
        import pilot_monitor_app
        classification, path = self.setup_task()
        regression, _ = self.setup_task("regression")
        workbook = self.root / "data.xlsx"
        frame = pd.DataFrame(dict(Subject=[1, 1], level=[1, 4], run=[1, 2],
            flight_hours=[10, 10], performance=[12., 14.], HRV_SDNN=[1.5, 2.3], scl_mean=[1.2, .4]))
        frame.to_excel(workbook, sheet_name="data", index=False)
        with patch.object(pilot_monitor_app, "RESULTS", self.root):
            app = pilot_monitor_app.create_app(classification, regression, workbook, self.root / "missing_raw")
            client = app.test_client()
            first = client.get("/api/evidence?task=classification&row=2")
            self.assertEqual(first.status_code, 200)
            self.assertFalse(first.json["evidence"]["shap_available"])
            self.explain("classification", path)
            fresh = client.get("/api/evidence?task=classification&row=2")
            self.assertTrue(fresh.json["evidence"]["shap_available"])
            self.assertFalse(fresh.json["evidence"]["shap_metadata"]["is_approximation"])
            self.assertEqual(len(client.get("/api/shap/cases?task=classification").json), 2)
            png = client.get("/api/shap/waterfall.png?task=classification&row=2")
            self.assertEqual(png.status_code, 200)
            self.assertTrue(png.data.startswith(b"\x89PNG"))
            self.assertEqual(client.get("/api/shap/values.csv?task=classification&row=2").status_code, 200)
            with patch("narrative_explainer._llm_paraphrase", return_value=("Saved model SHAP evidence.",
                {"used": True, "requested_model": "test", "resolved_model": "test"})) as mocked:
                a = client.get("/api/evidence?task=classification&row=2&llm=1")
                b = client.get("/api/evidence?task=classification&row=2&llm=1")
                self.assertEqual(a.status_code, 200)
                self.assertTrue(b.json["narrative"]["cache_hit"])
                offline = client.get("/api/evidence?task=classification&row=2")
                self.assertTrue(offline.json["narrative"]["cache_hit"])
                self.assertEqual(offline.json["narrative"]["llm_paraphrase"], "Saved model SHAP evidence.")
                self.assertEqual(mocked.call_count, 1)
            self.assertEqual(client.get("/api/evidence?task=bad&row=2").status_code, 400)

    def test_llm_rejection_keeps_deterministic_card(self):
        import narrative_explainer
        evidence = {"task": "classification", "source_excel_row": 2}
        deterministic = {"headline": "Evidence", "summary": "Saved numbers", "drivers": []}
        with patch.object(narrative_explainer, "deterministic_narrative", return_value=deterministic), patch.object(
            narrative_explainer, "_llm_paraphrase", return_value=(None, {"used": False,
                "reason": "LLM output failed grounding gate: invented number", "checks": {"numbers_in_evidence": False}})):
            result = stored_narrative(evidence, True, self.root)
            self.assertEqual(result["summary"], "Saved numbers")
            self.assertIsNone(result["llm_paraphrase"])
            self.assertEqual(result["llm_audit"]["status"], "gate_rejected")

if __name__ == "__main__":
    unittest.main()
