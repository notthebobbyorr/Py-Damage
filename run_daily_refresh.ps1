# run_daily_refresh.ps1
# Wrapper for Windows Task Scheduler - activates venv and runs the daily pipeline.
# Logs are written to data/logs/refresh_YYYY-MM-DD.log

$RepoDir  = $PSScriptRoot
$Python   = Join-Path $RepoDir ".venv\Scripts\python.exe"
$Script   = Join-Path $RepoDir "run_daily_refresh.py"
$LogsDir  = Join-Path $RepoDir "data\logs"
$LogFile  = Join-Path $LogsDir ("refresh_" + (Get-Date -Format "yyyy-MM-dd") + ".log")

# Ensure logs directory exists
if (-not (Test-Path $LogsDir)) { New-Item -ItemType Directory -Path $LogsDir | Out-Null }

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Write-Log "=== Daily refresh started ==="
Write-Log "Repo:   $RepoDir"
Write-Log "Python: $Python"
Write-Log "Log:    $LogFile"

if (-not (Test-Path $Python)) {
    Write-Log "ERROR: Python not found at $Python - aborting."
    exit 1
}

Set-Location $RepoDir

& $Python $Script 2>&1 | ForEach-Object {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $_
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

$ExitCode = $LASTEXITCODE
Write-Log "=== Daily refresh finished (exit code: $ExitCode) ==="
exit $ExitCode
