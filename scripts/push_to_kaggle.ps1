param (
    [string]$KernelPath = (Get-Item $PSScriptRoot\..).FullName
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ">>> Pushing OpenMOSS LoRA Training Pipeline to Kaggle <<<" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$metaFile = Join-Path $KernelPath "kernel-metadata.json"
if (-not (Test-Path $metaFile)) {
    Write-Host "[ERROR] kernel-metadata.json not found at: $metaFile" -ForegroundColor Red
    exit 1
}

$meta = Get-Content $metaFile -Raw -Encoding UTF8 | ConvertFrom-Json
$kernelId = $meta.id
if (-not $kernelId -or $kernelId -match "YOUR_KAGGLE_USERNAME") {
    Write-Host "[ERROR] Please update 'id' in kernel-metadata.json with your own Kaggle username first!" -ForegroundColor Red
    Write-Host "        Example: `"id`": `"your_kaggle_name/moss-image-lora-training`"" -ForegroundColor Yellow
    exit 1
}

if ($meta.machine_shape -and $meta.machine_shape.ToString().Trim()) {
    kaggle kernels push -p "$KernelPath" --accelerator $meta.machine_shape.ToString().Trim()
} else {
    kaggle kernels push -p "$KernelPath"
}
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nKernel ($kernelId) successfully pushed! Monitoring status..." -ForegroundColor Green
    kaggle kernels status $kernelId
} else {
    Write-Host "`nError pushing kernel to Kaggle." -ForegroundColor Red
}
