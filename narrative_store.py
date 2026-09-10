"""Persist checked summaries by evidence, prompt version and requested model."""
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from direct_shap import atomic_json
from narrative_explainer import PROMPT_VERSION, verbalize_evidence

LOCK = threading.Lock()

def stored_narrative(evidence, use_llm, directory):
    identity = dict(evidence=evidence, prompt_version=PROMPT_VERSION,
                    requested_model=os.environ.get("OPENROUTER_MODEL", "openrouter/free"),
                    base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    directory = Path(directory)
    path = directory / "narratives" / f"{key}.json"
    # Serialize requests so duplicate browser requests do not spend two calls.
    with LOCK:
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            result["cache_hit"] = True
            return result
        if not use_llm:
            return verbalize_evidence(evidence, use_llm=False)
        result = verbalize_evidence(evidence, use_llm=True)
        result["cache_hit"] = False
        audit = result["llm_audit"]
        reason = audit.get("reason", "")
        result["llm_audit"]["status"] = (
            "accepted" if audit.get("used") else "gate_rejected" if "grounding gate" in reason
            else "request_failed" if "request failed" in reason else "unavailable")
        record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
            task=evidence["task"], source_excel_row=evidence["source_excel_row"],
            prompt_version=result["prompt_version"], evidence_sha256=result["evidence_sha256"],
            requested_model=identity["requested_model"], llm_audit=result["llm_audit"],
            llm_paraphrase=result.get("llm_paraphrase"))
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "llm_narration_audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        if audit.get("used") or "grounding gate" in reason:
            atomic_json(path, result)
        return result
