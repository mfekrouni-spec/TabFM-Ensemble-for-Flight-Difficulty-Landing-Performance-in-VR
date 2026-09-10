#!/usr/bin/env python
"""Grounded pilot-facing narration of saved TabFM OOF predictions and SHAP.

The deterministic evidence card is always produced first.  An optional LLM
may paraphrase that already-validated card, but it cannot change the model
prediction, calculate new values, or make physiological/causal claims.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


from workflow_paths import ROOT as DEFAULT_ROOT, WORKBOOK as DEFAULT_WORKBOOK, task_dir, CONFIG
os.environ.setdefault("OPENROUTER_MODEL", CONFIG["openrouter_model"])
DEFAULT_CLASSIFICATION = task_dir("classification")
DEFAULT_REGRESSION = task_dir("regression")
PROMPT_VERSION = "pilot-grounded-narrator-v4-direct-all-runs"


def _plain_feature_label(feature: str, modality: str) -> str:
    """Translate engineered feature identifiers without adding physiology."""
    value = str(feature)
    wavelet = re.match(
        r"(.+)_wavelet_(energy|entropy)_level_(\d+)(?:_x)?$", value
    )
    if wavelet:
        source, statistic, level = wavelet.groups()
        source_labels = {
            "forearm_magnitude": "combined forearm-acceleration magnitude",
            "accelerometry_forearm_r_x_mps2": (
                "right-forearm acceleration on the x axis"
            ),
            "accelerometry_forearm_r_y_mps2": (
                "right-forearm acceleration on the y axis"
            ),
            "accelerometry_forearm_r_z_mps2": (
                "right-forearm acceleration on the z axis"
            ),
            "accelerometry_torso_x_mps2": "torso acceleration on the x axis",
            "accelerometry_torso_y_mps2": "torso acceleration on the y axis",
            "accelerometry_torso_z_mps2": "torso acceleration on the z axis",
        }
        source_label = source_labels.get(
            source, source.replace("_", " ")
        )
        statistic_label = (
            "signal energy" if statistic == "energy" else "signal complexity"
        )
        return f"{statistic_label} at wavelet level {level} of {source_label}"
    spectral_channel = re.fullmatch(r"psd_max_(.+)", value)
    if spectral_channel:
        return (
            "maximum power-spectral-density value of the "
            f"{spectral_channel.group(1)} eye-movement channel"
        )
    exact = {
        "flexor_median_amplitude": "median amplitude of forearm flexor EMG",
        "flexor_mean_frequency": "mean frequency of forearm flexor EMG",
        "spectral_centroid_y": "spectral centroid of the respiration y channel",
        "spectral_centroid_x": "spectral centroid of the respiration x channel",
        "phasic_mean": "mean value of the phasic EDA component",
    }
    if value in exact:
        return exact[value]
    return f"{modality} feature: {value.replace('_', ' ')}"


def plain_feature_label(feature: str, modality: str) -> str:
    """Public label helper shared by the monitor's local SHAP plot."""
    return _plain_feature_label(feature, modality)


_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?"
)


def _numeric_tokens(text: str) -> set[Decimal]:
    values: set[Decimal] = set()
    for token in _NUMBER_PATTERN.findall(text):
        try:
            values.add(Decimal(token.replace(",", "")))
        except InvalidOperation:
            continue
    return values


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _load_prediction(task_dir: Path, source_row: int, pd: Any) -> dict[str, Any]:
    path = task_dir / "all_models_oof_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"OOF prediction file not found: {path}")
    frame = pd.read_csv(path)
    match = frame.loc[
        frame["model"].astype(str).eq("tabfm_ensemble")
        & frame["source_excel_row"].astype(int).eq(int(source_row))
    ]
    if len(match) != 1:
        raise ValueError(
            f"Expected one TabFM OOF prediction for Excel row {source_row}; "
            f"found {len(match)}."
        )
    return match.iloc[0].to_dict()


def _load_shap_rows(task_dir, source_row, pd):
    from direct_shap import meta_path, validate_group
    prediction = _load_prediction(task_dir, source_row, pd)
    path = task_dir / "models" / "tabfm_ensemble" / "folds" / f"fold_{int(prediction['fold']):02d}_shap_values.csv"
    if not path.is_file():
        return pd.DataFrame()
    if not meta_path(path).is_file():
        raise ValueError("This SHAP file has no verification metadata; run PRECOMPUTE_ALL_SHAP.ps1 in the new folder")
    metadata = json.loads(meta_path(path).read_text(encoding="utf-8"))
    if metadata["ensemble_members"] != 32:
        raise ValueError("The GUI requires the full 32-member ensemble")
    frame = pd.read_csv(path)
    selected = frame.loc[frame.source_excel_row.eq(int(source_row))].copy()
    if selected.empty:
        return selected
    validate_group(selected, metadata, prediction)
    selected = selected.sort_values("shap_value", key=lambda x: x.abs(), ascending=False)
    selected["_shap_source"] = "direct_full_ensemble"
    selected["_is_approximation"] = False
    selected["_fidelity_r2"] = float("nan")
    return selected


def load_shap_rows(task_dir: Path, source_row: int) -> Any:
    """Load verified direct full-ensemble SHAP only."""
    import pandas as pd

    return _load_shap_rows(task_dir, source_row, pd)


@lru_cache(maxsize=4)
def _workbook_modality_profile(workbook_path: str) -> tuple[Any, Any, Any, list[str]]:
    import numpy as np
    import pandas as pd

    from tabfm_dual_task_optimized import assign_modality

    frame = pd.read_excel(workbook_path, sheet_name="data")
    predictors = frame.iloc[:, 5:].apply(pd.to_numeric, errors="coerce")
    medians = predictors.median(axis=0)
    scales = predictors.std(axis=0, ddof=1).replace(0, np.nan)
    modalities = [assign_modality(value) for value in predictors.columns]
    return predictors, medians, scales, modalities


def _modality_snapshot(
    workbook: Path,
    source_row: int,
    pd: Any,
    np: Any,
) -> list[dict[str, Any]]:
    predictors, medians, scales, modalities = _workbook_modality_profile(
        str(workbook.resolve())
    )
    position = int(source_row) - 2
    if position < 0 or position >= len(predictors):
        raise ValueError(f"Excel row {source_row} is outside the data sheet.")
    z_values = ((predictors.iloc[position] - medians) / scales).replace(
        [np.inf, -np.inf], np.nan
    )
    modality_frame = pd.DataFrame(
        {
            "feature": predictors.columns.astype(str),
            "modality": modalities,
            "absolute_standardized_deviation": z_values.abs().to_numpy(float),
        }
    )
    summary = (
        modality_frame.groupby("modality", as_index=False)
        .agg(
            mean_absolute_standardized_deviation=(
                "absolute_standardized_deviation",
                "mean",
            ),
            available_features=("feature", "count"),
        )
        .sort_values("mean_absolute_standardized_deviation", ascending=False)
    )
    return summary.to_dict(orient="records")


def build_evidence(
    task: str,
    source_row: int,
    task_dir: Path,
    workbook: Path = DEFAULT_WORKBOOK,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    prediction = _load_prediction(task_dir, source_row, pd)
    shap_rows = _load_shap_rows(task_dir, source_row, pd)
    drivers: list[dict[str, Any]] = []
    shap_decomposition: dict[str, Any] | None = None
    shap_metadata: dict[str, Any] | None = None
    if not shap_rows.empty:
        base_value = float(shap_rows.iloc[0]["base_value"])
        shap_sum = float(shap_rows["shap_value"].astype(float).sum())
        output_column = (
            "probability_hard"
            if task == "classification"
            else "predicted_performance"
        )
        if output_column in shap_rows.columns:
            explained_output = float(shap_rows.iloc[0][output_column])
            full_ensemble_output = explained_output
        elif "fast_tabfm_output" in shap_rows.columns:
            explained_output = float(shap_rows.iloc[0]["fast_tabfm_output"])
            full_ensemble_output = float(
                shap_rows.iloc[0].get(
                    "full_ensemble_output", prediction[output_column]
                )
            )
        else:
            raise ValueError(
                f"Saved SHAP rows do not contain a recognized {task} output."
            )
        is_approximation = bool(shap_rows.iloc[0]["_is_approximation"])
        fidelity_raw = shap_rows.iloc[0]["_fidelity_r2"]
        fidelity_r2 = (
            float(fidelity_raw) if pd.notna(fidelity_raw) else None
        )
        method = str(
            shap_rows.iloc[0].get("explanation_method", "saved SHAP")
        )
        shap_metadata = {
            "method": method,
            "source": str(shap_rows.iloc[0]["_shap_source"]),
            "is_approximation": is_approximation,
            "fidelity_r2": fidelity_r2,
            "feature_count": int(len(shap_rows)),
            "direct_full_ensemble_preferred": True,
        }
        shap_decomposition = {
            "baseline_output": base_value,
            "net_shap_shift": shap_sum,
            "reconstructed_output": base_value + shap_sum,
            "explained_model_output": explained_output,
            "reconstruction_difference": abs(
                (base_value + shap_sum) - explained_output
            ),
            "full_ensemble_output": full_ensemble_output,
            "approximation_difference": (
                abs(explained_output - full_ensemble_output)
                if is_approximation
                else 0.0
            ),
        }
        for rank, row in enumerate(
            shap_rows.head(12).to_dict(orient="records"), start=1
        ):
            shap_value = float(row["shap_value"])
            modality = str(row.get("modality", "Other"))
            effect_value = (
                100.0 * shap_value
                if task == "classification"
                else shap_value
            )
            drivers.append(
                {
                    "rank_by_absolute_shap": rank,
                    "feature": str(row["feature"]),
                    "plain_feature": _plain_feature_label(
                        str(row["feature"]), modality
                    ),
                    "modality": modality,
                    "feature_value": float(row["feature_value"]),
                    "shap_value": shap_value,
                    "direction": "increased" if shap_value > 0 else "decreased",
                    "effect_value": effect_value,
                    "effect_unit": (
                        "percentage points of hard-class probability"
                        if task == "classification"
                        else "predicted Performance target units"
                    ),
                }
            )
    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "task": task,
        "model": "tabfm_ensemble",
        "model_display_name": "TabFM Ensemble",
        "source_excel_row": int(source_row),
        "subject": prediction.get("subject"),
        "run": prediction.get("run"),
        "level": prediction.get("level"),
        "flight_hours": prediction.get("flight_hours"),
        "fold": prediction.get("fold"),
        "shap_available": bool(drivers),
        "top_shap_drivers": drivers,
        "shap_decomposition": shap_decomposition,
        "shap_metadata": shap_metadata,
        "modality_snapshot": _modality_snapshot(
            workbook, source_row, pd, np
        ),
        "scope": (
            "Out-of-fold prediction from the sample-level paper protocol; "
            "pilots may overlap between training and test folds."
        ),
    }
    if task == "classification":
        evidence["prediction"] = {
            "observed_class": str(prediction["actual_class"]),
            "predicted_class": str(prediction["predicted_class"]),
            "probability_easy": float(prediction["probability_easy"]),
            "probability_hard": float(prediction["probability_hard"]),
            "correct": str(prediction["correct"]).casefold() == "true",
            "member_probability_hard_sd": (
                float(
                    prediction[
                        "ensemble_member_probability_hard_std_uncalibrated"
                    ]
                )
                if "ensemble_member_probability_hard_std_uncalibrated"
                in prediction
                else None
            ),
        }
        evidence["shap_output_definition"] = (
            "Positive SHAP values increase the hard-class probability; "
            "negative values decrease it. For fast representative cases, "
            "the contributions explain the audited TabFM approximation, "
            "not the weighted/calibrated 32-member ensemble directly."
            if shap_metadata and shap_metadata["is_approximation"]
            else "Positive SHAP values increase the model probability of "
            "hard; negative values decrease it."
        )
    else:
        evidence["prediction"] = {
            "observed_performance": float(prediction["actual_performance"]),
            "predicted_performance": float(
                prediction["predicted_performance"]
            ),
            "residual_observed_minus_predicted": float(prediction["residual"]),
            "absolute_error": float(prediction["absolute_error"]),
            "member_prediction_sd": (
                float(prediction["ensemble_member_prediction_std"])
                if "ensemble_member_prediction_std" in prediction
                else None
            ),
        }
        evidence["shap_output_definition"] = (
            "Positive SHAP values increase predicted performance in target "
            "units; negative values decrease it. For fast representative "
            "cases, the contributions explain the audited TabFM "
            "approximation, not the weighted/calibrated 32-member ensemble "
            "directly."
            if shap_metadata and shap_metadata["is_approximation"]
            else "Positive SHAP values increase predicted performance in "
            "target units; negative values decrease it."
        )
    return evidence


def deterministic_narrative(evidence: dict[str, Any]) -> dict[str, Any]:
    prediction = evidence["prediction"]
    if evidence["task"] == "classification":
        predicted = prediction["predicted_class"].title()
        hard_probability = prediction["probability_hard"]
        headline = f"Predicted flight difficulty: {predicted}"
        summary = (
            f"TabFM assigned {hard_probability:.1%} probability to the hard "
            "class for this run."
        )
    else:
        predicted = prediction["predicted_performance"]
        headline = f"Predicted landing performance: {predicted:,.1f}"
        summary = (
            "This is the held-out TabFM estimate for the selected run. "
            f"Its absolute OOF error was {prediction['absolute_error']:,.1f}."
        )
    drivers = evidence["top_shap_drivers"]
    if drivers:
        increasing = [value for value in drivers if value["shap_value"] > 0][:3]
        decreasing = [value for value in drivers if value["shap_value"] < 0][:3]
        driver_text = []
        decomposition = evidence.get("shap_decomposition") or {}
        metadata = evidence.get("shap_metadata") or {}
        if metadata.get("is_approximation"):
            fidelity = metadata.get("fidelity_r2")
            fidelity_text = (
                f" (OOF output fidelity R²={fidelity:.3f})"
                if fidelity is not None
                else ""
            )
            driver_text.append(
                "This local explanation uses the fast TabFM approximation"
                f"{fidelity_text}; the displayed ensemble prediction remains "
                "the saved 32-member OOF output."
            )
        if evidence["task"] == "classification" and decomposition:
            driver_text.append(
                "SHAP started from a hard-class baseline of "
                f"{100 * decomposition['baseline_output']:.1f}%. Together, "
                "the saved feature contributions shifted the output by "
                f"{100 * decomposition['net_shap_shift']:+.1f} percentage "
                "points, reconstructing approximately "
                f"{100 * decomposition['reconstructed_output']:.1f}%."
            )
        elif decomposition:
            driver_text.append(
                "SHAP started from a predicted-Performance baseline of "
                f"{decomposition['baseline_output']:,.1f}. Together, the "
                "saved feature contributions shifted the prediction by "
                f"{decomposition['net_shap_shift']:+,.1f} target units."
            )
        if increasing:
            driver_text.append(
                (
                    "Moved the model toward Hard: "
                    if evidence["task"] == "classification"
                    else "Raised predicted Performance: "
                )
                + ", ".join(
                    f"{value['plain_feature']} ({value['modality']}; "
                    f"approximately {value['effect_value']:+.1f} "
                    f"{value['effect_unit']})"
                    for value in increasing
                )
            )
        if decreasing:
            driver_text.append(
                (
                    "Moved the model toward Easy: "
                    if evidence["task"] == "classification"
                    else "Lowered predicted Performance: "
                )
                + ", ".join(
                    f"{value['plain_feature']} ({value['modality']}; "
                    f"approximately {value['effect_value']:+.1f} "
                    f"{value['effect_unit']})"
                    for value in decreasing
                )
            )
    else:
        driver_text = [
            "Run-level SHAP is not available yet. The displayed modality "
            "profile is descriptive and is not a model explanation."
        ]
    return {
        "headline": headline,
        "summary": summary,
        "drivers": driver_text,
        "caution": (
            "SHAP describes this model decision, not a physiological cause or "
            "clinical state. The result is not validated for unseen pilots."
        ),
    }


def _detailed_llm_payload(evidence: dict[str, Any]) -> dict[str, Any]:
    prediction = evidence["prediction"]
    decomposition = evidence.get("shap_decomposition") or {}
    metadata = evidence.get("shap_metadata") or {}
    drivers = evidence["top_shap_drivers"]
    increasing = [value for value in drivers if value["shap_value"] > 0][:4]
    decreasing = [value for value in drivers if value["shap_value"] < 0][:4]

    def driver_payload(value: dict[str, Any]) -> dict[str, Any]:
        return {
            "rank_by_absolute_shap": int(value["rank_by_absolute_shap"]),
            "feature_id": value["feature"],
            "plain_feature": value["plain_feature"],
            "modality": value["modality"],
            "signed_shap_effect": round(float(value["effect_value"]), 1),
            "absolute_shap_effect": round(
                abs(float(value["effect_value"])), 1
            ),
            "effect_unit": value["effect_unit"],
        }

    payload: dict[str, Any] = {
        "task": evidence["task"],
        "model": evidence["model_display_name"],
        "explanation_method": metadata.get("method", "saved SHAP"),
        "output_definition": evidence["shap_output_definition"],
        "interpretation_rule": (
            "Each SHAP effect is an approximate contribution to this saved "
            "model output relative to the SHAP baseline. It is not a change "
            "in the sensor signal and is not a physiological cause."
        ),
        "drivers_that_increased_output": [
            driver_payload(value) for value in increasing
        ],
        "drivers_that_decreased_output": [
            driver_payload(value) for value in decreasing
        ],
    }
    if metadata.get("is_approximation"):
        payload["approximation_audit"] = {
            "local_values_explain_fast_tabfm_proxy": True,
            "full_ensemble_prediction_is_reference_only": True,
            "oof_output_fidelity_r2": metadata.get("fidelity_r2"),
            "fast_tabfm_output": round(
                float(decomposition.get("explained_model_output", 0.0)), 4
            ),
            "full_ensemble_output": round(
                float(decomposition.get("full_ensemble_output", 0.0)), 4
            ),
        }
    if evidence["task"] == "classification":
        payload["prediction"] = {
            "predicted_class": prediction["predicted_class"].title(),
            "observed_class": prediction["observed_class"].title(),
            "hard_probability_percent": round(
                100.0 * float(prediction["probability_hard"]), 1
            ),
            "correct_oof_classification": bool(prediction["correct"]),
        }
        payload["shap_decomposition"] = {
            "baseline_hard_probability_percent": round(
                100.0 * float(decomposition["baseline_output"]), 1
            ),
            "net_shap_shift_percentage_points": round(
                100.0 * float(decomposition["net_shap_shift"]), 1
            ),
            "shap_reconstructed_hard_probability_percent": round(
                100.0 * float(decomposition["reconstructed_output"]), 1
            ),
        }
        payload["direction_labels"] = {
            "positive": "toward Hard",
            "negative": "toward Easy",
        }
    else:
        payload["prediction"] = {
            "predicted_performance": round(
                float(prediction["predicted_performance"]), 1
            ),
            "observed_performance": round(
                float(prediction["observed_performance"]), 1
            ),
            "absolute_oof_error": round(
                float(prediction["absolute_error"]), 1
            ),
        }
        payload["shap_decomposition"] = {
            "baseline_predicted_performance": round(
                float(decomposition["baseline_output"]), 1
            ),
            "net_shap_shift_target_units": round(
                float(decomposition["net_shap_shift"]), 1
            ),
            "shap_reconstructed_prediction": round(
                float(decomposition["reconstructed_output"]), 1
            ),
        }
        payload["direction_labels"] = {
            "positive": "toward a higher predicted Performance value",
            "negative": "toward a lower predicted Performance value",
        }
    return payload


def _llm_paraphrase(
    evidence: dict[str, Any],
    deterministic: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None, {"used": False, "reason": "OPENROUTER_API_KEY is not set"}
    from openai import OpenAI

    model = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
    base_url = os.environ.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    if not evidence.get("shap_available"):
        return None, {
            "used": False,
            "reason": "Run-level SHAP is not available for detailed narration",
            "provider": "OpenRouter",
            "requested_model": model,
        }
    safe_payload = _detailed_llm_payload(evidence)
    instructions = (
        "You are the plain-language explanation layer of a post-session pilot "
        "training monitor. Use only the supplied JSON. Explain the saved SHAP "
        "evidence in five to eight short lines and no more than 180 words. "
        "Use this order: Model result; How to read SHAP; strongest evidence "
        "that increased the output; strongest evidence that decreased the "
        "output; overall reading. Name the plain-language features, their "
        "modalities, directions, and supplied approximate SHAP effects. For "
        "classification, describe effects as percentage-point shifts in the "
        "hard-class probability relative to the SHAP baseline. Make clear "
        "that a SHAP value is a model-output contribution, not a sensor-value "
        "change. Do not invent or calculate numbers. Do not infer workload, "
        "fatigue, stress, attention, health, performance mechanisms, causes, "
        "safety status, or recommendations. Do not call a feature good, bad, "
        "normal, or abnormal. Say 'the model' and 'the saved SHAP evidence'. "
        "Use plain text lines; no table and no technical feature identifiers."
    )
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=45.0, max_retries=0)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": json.dumps(
                    safe_payload, default=_json_safe, sort_keys=True
                ),
            },
        ],
        temperature=0,
    )
    text = (response.choices[0].message.content or "").strip()
    resolved_model = getattr(response, "model", None)
    banned = re.compile(
        r"diagnos|fatigue|stress|workload|attention|medical|health|unsafe|"
        r"impair|recommend|\bshould\b|\bmust\b|\bnormal\b|\babnormal\b",
        re.IGNORECASE,
    )
    causal_check = re.sub(
        r"\b(?:does not|is not|not)\s+(?:imply\s+)?caus(?:e|al|ation)\b",
        "",
        text.casefold(),
    )
    claim_violation = bool(
        banned.search(text)
        or re.search(r"\bcaus(?:e|ed|es|al|ation)\b", causal_check)
    )
    payload_text = json.dumps(safe_payload, default=_json_safe, sort_keys=True)
    invented_numbers = sorted(
        _numeric_tokens(text).difference(_numeric_tokens(payload_text))
    )
    missing_core = (
        "shap" not in text.casefold()
        or (
            evidence["task"] == "classification"
            and str(evidence["prediction"]["predicted_class"]).casefold()
            not in text.casefold()
        )
    )
    checks = {"nonempty": bool(text), "length": len(text) <= 2400,
              "no_prohibited_claim": not claim_violation,
              "numbers_in_evidence": not bool(invented_numbers), "core_context": not missing_core}
    if (
        not text
        or len(text) > 2400
        or claim_violation
        or invented_numbers
        or missing_core
    ):
        reasons = []
        if not text:
            reasons.append("empty output")
        if len(text) > 2400:
            reasons.append("output too long")
        if claim_violation:
            reasons.append("disallowed physiological/causal/prescriptive claim")
        if invented_numbers:
            reasons.append(
                "numbers not present in evidence: "
                + ", ".join(str(value) for value in invented_numbers)
            )
        if missing_core:
            reasons.append("prediction or SHAP context missing")
        return None, {
            "used": False,
            "reason": "LLM output failed grounding gate: " + "; ".join(reasons),
            "checks": checks,
            "rejected_text": text,
            "provider": "OpenRouter",
            "requested_model": model,
            "resolved_model": resolved_model,
        }
    return text, {
        "used": True,
        "checks": checks,
        "provider": "OpenRouter",
        "requested_model": model,
        "resolved_model": resolved_model,
    }


def verbalize_evidence(
    evidence: dict[str, Any], use_llm: bool = False
) -> dict[str, Any]:
    deterministic = deterministic_narrative(evidence)
    payload_text = json.dumps(evidence, default=_json_safe, sort_keys=True)
    result = {
        **deterministic,
        "prompt_version": PROMPT_VERSION,
        "evidence_sha256": hashlib.sha256(payload_text.encode()).hexdigest(),
        "llm_paraphrase": None,
        "llm_audit": {"used": False, "reason": "LLM not requested"},
    }
    if use_llm:
        try:
            paraphrase, audit = _llm_paraphrase(evidence, deterministic)
            result["llm_paraphrase"] = paraphrase
            result["llm_audit"] = audit
        except Exception as exc:
            # The deterministic evidence card must remain available even when
            # the optional network-backed paraphrase cannot be produced.
            result["llm_audit"] = {
                "used": False,
                "reason": f"LLM request failed: {type(exc).__name__}. Check provider access and try again.",
                "provider": "OpenRouter",
                "requested_model": os.environ.get(
                    "OPENROUTER_MODEL", "openrouter/free"
                ),
            }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    parser.add_argument("--source-row", type=int, required=True)
    parser.add_argument("--task-dir", type=Path, default=None)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_dir = args.task_dir or (
        DEFAULT_CLASSIFICATION
        if args.task == "classification"
        else DEFAULT_REGRESSION
    )
    evidence = build_evidence(
        args.task, args.source_row, task_dir.resolve(), args.workbook.resolve()
    )
    narrative = verbalize_evidence(evidence, use_llm=args.use_llm)
    result = {"evidence": evidence, "narrative": narrative}
    rendered = json.dumps(result, indent=2, default=_json_safe)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Saved grounded narrative: {args.output.resolve()}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
