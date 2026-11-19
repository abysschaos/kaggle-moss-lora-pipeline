param (
    [string]$Kernel = "",
    [string]$OutputDir = "$PSScriptRoot\..\output"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ">>> Pulling LoRA Weights & Checkpoints from Kaggle <<<" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if (-not $Kernel) {
    $metaFile = Join-Path $PSScriptRoot "..\kernel-metadata.json"
    if (Test-Path $metaFile) {
        $meta = Get-Content $metaFile -Raw -Encoding UTF8 | ConvertFrom-Json
        $Kernel = $meta.id
    }
}

if (-not $Kernel -or $Kernel -match "YOUR_KAGGLE_USERNAME") {
    Write-Host "[ERROR] Please pass -Kernel 'username/kernel-slug' or configure 'id' in kernel-metadata.json!" -ForegroundColor Red
    exit 1
}

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
