# ==============================================================================
# OpenMOSS MOSS-Image Local GUI Launcher (PowerShell)
# ==============================================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $scriptDir "..")

$pythonExe = ""
if (Test-Path ".\venv\Scripts\python.exe") {
    $pythonExe = (Resolve-Path ".\venv\Scripts\python.exe").Path
} elseif ($env:COMFYUI_PYTHON -and (Test-Path $env:COMFYUI_PYTHON)) {
    $pythonExe = $env:COMFYUI_PYTHON
} else {
    $detectedPy = (python ".\scripts\detect_env.py" 2>$null | Select-Object -First 1)
    if ($detectedPy -and (Test-Path $detectedPy)) {
        $pythonExe = $detectedPy
    }
}

if (-not $pythonExe) {
    $pythonExe = "python"
}

Write-Host "Launching OpenMOSS Local LoRA Trainer GUI with: $pythonExe" -ForegroundColor Green
& $pythonExe gui_local.py
