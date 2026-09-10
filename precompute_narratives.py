"""Optional manual batch generation of checked LLM summaries after direct SHAP."""
import argparse
import json
import os
from collections import Counter

from direct_shap import atomic_json, collect_validated
from narrative_explainer import build_evidence
from narrative_store import stored_narrative
from workflow_paths import RESULTS, WORKBOOK, CONFIG, task_dir

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression", "both"), default="both")
    parser.add_argument("--limit", type=int, help="Optional number of cases per task for a first manual trial")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("Set OPENROUTER_API_KEY in this process or use PRECOMPUTE_LLM.ps1")
    os.environ.setdefault("OPENROUTER_MODEL", CONFIG["openrouter_model"])
    for task in ("classification", "regression") if args.task == "both" else (args.task,):
        values, coverage = collect_validated(task_dir(task))
        if not coverage["complete"]:
            raise RuntimeError(f"Complete direct SHAP for {task} first: {coverage['direct_rows']}/{coverage['total_rows']}")
        rows = sorted(values.source_excel_row.unique().astype(int))
        if args.limit:
            rows = rows[:args.limit]
        counts = Counter()
        for index, row in enumerate(rows, 1):
            evidence = build_evidence(task, int(row), task_dir(task), WORKBOOK)
            result = stored_narrative(evidence, True, RESULTS)
            status = result["llm_audit"]["status"]
            counts[status] += 1
            counts["cache_hits"] += int(result.get("cache_hit", False))
            atomic_json(task_dir(task) / "llm_case_summary.json", dict(
                task=task, requested_cases=len(rows), processed_cases=index, counts=dict(counts),
                denominator="One result per requested task/run in this invocation; cached results included"))
            print(f"{task} {index}/{len(rows)} Excel row {row}: {status}", flush=True)
            if status in ("request_failed", "unavailable"):
                raise RuntimeError("LLM unavailable; saved prior work. Resolve provider access and rerun to resume.")
        print(json.dumps(dict(counts)), flush=True)

if __name__ == "__main__":
    main()
