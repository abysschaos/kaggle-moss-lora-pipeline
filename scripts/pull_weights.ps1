param (
    [string]$Kernel = "norrisgavincortni/ness-style-lora-training",
    [string]$OutputDir = "$PSScriptRoot\..\output"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ">>> Pulling LoRA Weights & Checkpoints from Kaggle <<<" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
}

kaggle kernels output $Kernel -p "$OutputDir"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nWeights downloaded successfully to: $OutputDir" -ForegroundColor Green
    Get-ChildItem -Recurse -Path "$OutputDir" -Filter "*.safetensors" | Select-Object Name, Length, LastWriteTime
} else {
    Write-Host "`nFailed to download kernel output. Ensure kernel status is COMPLETE." -ForegroundColor Red
}
