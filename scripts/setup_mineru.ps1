param(
    [string]$Python = "python",
    [switch]$ForceRecreate
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = Join-Path $projectRoot ".venv-mineru"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if ($ForceRecreate -and (Test-Path -LiteralPath $venvPath)) {
    $resolvedVenv = (Resolve-Path -LiteralPath $venvPath).Path
    if ($resolvedVenv -ne $venvPath) {
        throw "Unexpected MinerU environment path: $resolvedVenv"
    }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create MinerU virtual environment."
    }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

& $venvPython -m pip install -r (
    Join-Path $projectRoot "requirements-mineru.txt"
)
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install MinerU."
}

$mineruExe = Join-Path $venvPath "Scripts\mineru.exe"
if (-not (Test-Path -LiteralPath $mineruExe)) {
    throw "MinerU CLI was not created at $mineruExe"
}

& $mineruExe --version
if ($LASTEXITCODE -ne 0) {
    throw "MinerU CLI verification failed."
}

Write-Host "MinerU environment ready: $venvPath"
