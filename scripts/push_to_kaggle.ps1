param (
    [string]$KernelPath = (Get-Item $PSScriptRoot\..).FullName
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ">>> Pushing OpenMOSS LoRA Training Pipeline to Kaggle <<<" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

kaggle kernels push -p "$KernelPath"
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nKernel successfully pushed! Monitoring status..." -ForegroundColor Green
    kaggle kernels status norrisgavincortni/ness-style-lora-training
} else {
    Write-Host "`nError pushing kernel to Kaggle." -ForegroundColor Red
}
