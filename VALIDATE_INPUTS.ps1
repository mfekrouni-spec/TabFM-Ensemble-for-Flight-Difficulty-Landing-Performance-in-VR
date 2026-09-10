. (Join-Path $PSScriptRoot "workflow_common.ps1")
& $workflowPython -X utf8 (Join-Path $workflowRoot "run_benchmarks.py") --validate-only
if ($LASTEXITCODE -ne 0) { throw "Input validation failed." }
