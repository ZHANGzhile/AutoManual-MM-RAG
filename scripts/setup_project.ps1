param(
    [string]$Python = "python",
    [switch]$ForceRecreate
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if ($ForceRecreate -and (Test-Path -LiteralPath $venvPath)) {
    $resolvedVenv = (Resolve-Path -LiteralPath $venvPath).Path
    if ($resolvedVenv -ne $venvPath) {
        throw "Unexpected project environment path: $resolvedVenv"
    }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the project virtual environment."
    }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

& $venvPython -m pip install -r (
    Join-Path $projectRoot "requirements-demo.txt"
)
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install project dependencies."
}

Write-Host "Project environment ready: $venvPath"
Write-Host "Python: $venvPython"
