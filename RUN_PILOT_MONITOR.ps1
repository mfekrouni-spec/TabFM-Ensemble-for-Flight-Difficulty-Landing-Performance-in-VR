param([switch]$SkipLlmPrompt, [switch]$NoBrowser, [int]$Port = 8766)
. (Join-Path $PSScriptRoot "workflow_common.ps1")
if (-not $SkipLlmPrompt) { Read-WorkflowLlmKey }
$workflowArguments = @("-X", "utf8", (Join-Path $workflowRoot "pilot_monitor_app.py"), "--port", $Port)
if ($NoBrowser) { $workflowArguments += "--no-open-browser" }
& $workflowPython @workflowArguments
if ($LASTEXITCODE -ne 0) { throw "Monitor stopped with an error." }
