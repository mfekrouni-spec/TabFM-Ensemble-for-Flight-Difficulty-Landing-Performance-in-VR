param([ValidateSet("classification", "regression", "both")][string]$Task = "both", [switch]$ValidateOnly)
. (Join-Path $PSScriptRoot "workflow_common.ps1")
$workflowArguments = @("-X", "utf8", "-u", (Join-Path $workflowRoot "run_benchmarks.py"), "--task", $Task)
if ($ValidateOnly) { $workflowArguments += "--validate-only" }
& $workflowPython @workflowArguments
if ($LASTEXITCODE -ne 0) { throw "Benchmark failed. Inspect the error above." }
