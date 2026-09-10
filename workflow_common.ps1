$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$workflowRoot = $PSScriptRoot
$workflowConfig = Get-Content -LiteralPath (Join-Path $workflowRoot "workflow_config.json") -Raw | ConvertFrom-Json
$workflowOverride = $env:TABFM_CONFIG
if (-not $workflowOverride) { $workflowOverride = Join-Path $workflowRoot 'workflow_config.local.json' }
if ($env:TABFM_CONFIG -and -not (Test-Path -LiteralPath $workflowOverride)) {
    throw "TABFM_CONFIG does not exist: $workflowOverride"
}
if (Test-Path -LiteralPath $workflowOverride) {
    $overrideConfig = Get-Content -LiteralPath $workflowOverride -Raw | ConvertFrom-Json
    foreach ($property in $overrideConfig.PSObject.Properties) {
        $workflowConfig | Add-Member -MemberType NoteProperty -Name $property.Name -Value $property.Value -Force
    }
}
$workflowPython = $env:TABFM_PYTHON
if (-not $workflowPython) {
    $localPython = Join-Path $workflowRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $localPython) { $workflowPython = $localPython }
    else { $workflowPython = (Get-Command python -ErrorAction Stop).Source }
}
function Read-WorkflowLlmKey {
    if (-not $env:OPENROUTER_MODEL) { $env:OPENROUTER_MODEL = $workflowConfig.openrouter_model }
    if (-not $env:OPENROUTER_API_KEY) {
        Write-Host "Enter an OpenRouter API key for LLM summaries. The key is not saved."
        Write-Host "Press Enter to use deterministic evidence cards only."
        $workflowSecureKey = Read-Host "OpenRouter API key" -AsSecureString
        $workflowPlainKey = [System.Net.NetworkCredential]::new("", $workflowSecureKey).Password
        if ($workflowPlainKey) { $env:OPENROUTER_API_KEY = $workflowPlainKey }
        $workflowPlainKey = $null
    }
}
