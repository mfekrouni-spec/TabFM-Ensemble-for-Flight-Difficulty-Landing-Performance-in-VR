"""Count generation attempts, individual checks and fallbacks without exposing text."""
import json
from collections import Counter
from direct_shap import atomic_json
from workflow_paths import RESULTS

def summarize(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    statuses, failed_checks, checked, models = Counter(), Counter(), Counter(), Counter()
    for record in records:
        audit = record["llm_audit"]
        statuses[audit.get("status", "unknown")] += 1
        failed_checks.update(name for name, passed in audit.get("checks", {}).items() if not passed)
        checked.update(audit.get("checks", {}).keys())
        models[audit.get("resolved_model") or audit.get("requested_model") or "unavailable"] += 1
    return dict(attempts=len(records), unique_task_run_records=len({(r["task"], r["source_excel_row"]) for r in records}),
        statuses=dict(statuses), failed_checks=dict(failed_checks), resolved_or_requested_models=dict(models),
        acceptance_fraction=statuses["accepted"] / len(records) if records else None,
        fallback_attempts=sum(count for status, count in statuses.items() if status != "accepted"),
        fallback_fraction=(len(records) - statuses["accepted"]) / len(records) if records else None,
        checks={name: {"evaluated": count, "passed": count - failed_checks[name],
                      "failed": failed_checks[name], "pass_fraction": (count - failed_checks[name]) / count}
                for name, count in sorted(checked.items())},
        denominator="Logged generation attempts; cache hits are not new attempts. Failed checks may overlap.")

if __name__ == "__main__":
    result = summarize(RESULTS / "llm_narration_audit.jsonl")
    atomic_json(RESULTS / "llm_audit_summary.json", result)
    print(json.dumps(result, indent=2))
