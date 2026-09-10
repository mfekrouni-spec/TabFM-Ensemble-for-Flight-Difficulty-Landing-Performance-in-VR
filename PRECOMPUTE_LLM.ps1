param([ValidateSet("classification", "regression", "both")][string]$Task = "both", [int]$Limit = 0)
. (Join-Path $PSScriptRoot "workflow_common.ps1")
Read-WorkflowLlmKey
if (-not $env:OPENROUTER_API_KEY) { throw "An OpenRouter API key is required for this optional step." }
$workflowArguments = @("-X", "utf8", "-u", (Join-Path $workflowRoot "precompute_narratives.py"), "--task", $Task)
if ($Limit -gt 0) { $workflowArguments += @("--limit", $Limit) }
& $workflowPython @workflowArguments
if ($LASTEXITCODE -ne 0) { throw "LLM preparation stopped. Existing summaries are preserved; rerun to resume." }
