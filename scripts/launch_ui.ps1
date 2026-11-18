# ==============================================================================
# OpenMOSS MOSS-Image Local GUI Launcher (PowerShell)
# ==============================================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $scriptDir "..")

Write-Host "Launching OpenMOSS Local LoRA Trainer GUI..." -ForegroundColor Green
python gui_local.py
