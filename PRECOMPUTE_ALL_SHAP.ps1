param(
    [ValidateSet("classification", "regression", "both")][string]$Task = "both",
    [ValidateRange(1, 5)][int[]]$Fold
)
. (Join-Path $PSScriptRoot "workflow_common.ps1")
$workflowArguments = @("-X", "utf8", "-u", (Join-Path $workflowRoot "run_all_shap.py"), "--task", $Task)
if ($Fold) { $workflowArguments += "--fold"; $workflowArguments += $Fold }
Write-Host "Computing direct 32-member ensemble SHAP for all held-out runs."
Write-Host "This can take a long time. Each completed run is saved; rerun this command to resume."
Write-Host "Configuration: $workflowRoot\workflow_config.json"
& $workflowPython @workflowArguments
if ($LASTEXITCODE -ne 0) { throw "SHAP stopped. Inspect the error above; completed runs remain saved." }
Write-Host "Requested SHAP preparation completed. Launch RUN_PILOT_MONITOR.ps1."
