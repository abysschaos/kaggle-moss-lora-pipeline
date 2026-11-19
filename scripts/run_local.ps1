# ==============================================================================
# OpenMOSS MOSS-Image Windows Local Training Launcher
# Target Hardware: NVIDIA GeForce RTX 4070 Laptop / 4060 / 3060 (8GB - 16GB VRAM)
# Compatible with Windows PowerShell 5.1 and PowerShell 7+
# ==============================================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "OpenMOSS MOSS-Image Local LoRA Trainer"

$projectRoot = (Get-Item $PSScriptRoot\..).FullName
Set-Location $projectRoot

$appVersion = "1.2.2"
$versionFile = Join-Path $projectRoot "VERSION"
if (Test-Path $versionFile) {
    $rawVer = (Get-Content $versionFile -Raw -Encoding UTF8).Trim()
    if ($rawVer) { $appVersion = $rawVer }
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ">>> OpenMOSS MOSS-Image Local LoRA Training Launcher v$appVersion (Windows CLI) <<<" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. 自动发现可用的高性能 PyTorch Python (venv -> COMFYUI_PYTHON -> detect_env.py -> 系统 python)
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
    $pythonCmd = (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $pythonCmd) {
        Write-Host "[ERROR] Python is not installed or not in PATH! Please install Python 3.10 - 3.12." -ForegroundColor Red
        exit 1
    }
    $pythonExe = $pythonCmd.Source
}

Write-Host "[1/4] Verifying Python and PyTorch Environment ($pythonExe)..." -ForegroundColor Yellow
$diagScript = @"
import sys
import torch
print('Python: ' + sys.version.split()[0])
print('PyTorch: ' + torch.__version__)
print('CUDA Available: ' + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print('GPU Device: ' + torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'Total VRAM: {vram:.2f} GB')
    print('BF16 Supported: ' + str(torch.cuda.is_bf16_supported()))
"@

& $pythonExe -c $diagScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] PyTorch CUDA verification failed. Make sure PyTorch with CUDA is installed." -ForegroundColor Red
    exit 1
}

# 2. 检查默认路径配置
$defaultDataDir = ".\dataset"
if (-not (Test-Path $defaultDataDir) -and (Test-Path "$env:USERPROFILE\Pictures")) {
    $defaultDataDir = "$env:USERPROFILE\Pictures"
}
$outputDir = ".\output"

Write-Host "`n[2/4] Checking Local Dataset Assets..." -ForegroundColor Yellow
if (Test-Path $defaultDataDir) {
    $imgCount = (Get-ChildItem $defaultDataDir -Include *.jpg,*.jpeg,*.png,*.webp -File -ErrorAction SilentlyContinue).Count
    $txtCount = (Get-ChildItem $defaultDataDir -Filter *.txt -File -ErrorAction SilentlyContinue).Count
    Write-Host " [OK] Found Dataset at: $defaultDataDir ($imgCount images, $txtCount txt captions)" -ForegroundColor Green
} else {
    Write-Host " [WARNING] Default dataset folder not found at: $defaultDataDir" -ForegroundColor Yellow
}

# 3. 运行参数确认 (兼容 PowerShell 5.1 语法)
$defaultLoraName = "my_lora"
$defaultTriggerWord = ""
$defaultVramPreset = "auto"
$defaultResolution = "1024"
$defaultTrainMode = "fresh"

$triggerDisplay = if ($defaultTriggerWord) { $defaultTriggerWord } else { "(None, using original captions)" }

Write-Host "`n[3/4] Default Training Profile for Local GPU:" -ForegroundColor Cyan
Write-Host " - LoRA Name:      $defaultLoraName (output subfolder and weights prefix)"
Write-Host " - Trigger Word:   $triggerDisplay"
Write-Host " - Train Mode:     $defaultTrainMode (fresh / resume / init_from)"
Write-Host " - VRAM Preset:    $defaultVramPreset (auto / 6g / 8g / 12g / 16g)"
Write-Host " - Resolution:     ${defaultResolution}x${defaultResolution} (1024 / 768 / 512)"
Write-Host " - Precision:      BF16 (Auto-falls back to FP16 on Turing/Pascal GPUs)"
Write-Host " - Text Device:    CPU (prevents Qwen3.5-9B OOM on 8GB VRAM)"
Write-Host " - Steps / Epochs: 600 Steps / 6 Epochs (Repeats=10)"
Write-Host " - LoRA Dim/Alpha: Rank=32, Alpha=32.0 (Full 1.0 Strength)"
Write-Host " - Learning Rate:  1e-4"
Write-Host " - Output Path:    $outputDir\$defaultLoraName"
Write-Host " - Tip:            Need Multi-Subset / DoRA / LoHa / LoKr / Live Loss Chart? Run MOSS_LoRA_Trainer.exe or pass CLI flags."

$confirm = Read-Host "`nPress Enter to start training immediately (or type 'custom' to adjust arguments)"
if ($confirm -eq "custom") {
    $customName = Read-Host "Enter custom LoRA name / folder name [$defaultLoraName]"
    if ($customName) { $defaultLoraName = $customName }

    $customTrigger = Read-Host "Enter custom trigger word (e.g. makima_char) [None]"
    if ($customTrigger) { $defaultTriggerWord = $customTrigger }

    $customMode = Read-Host "Enter training mode (fresh / resume / init_from) [$defaultTrainMode]"
    if ($customMode) { $defaultTrainMode = $customMode }

    $customData = Read-Host "Enter dataset path [$defaultDataDir]"
    if ($customData) { $defaultDataDir = $customData }

    $customSteps = Read-Host "Enter training steps [600]"
    if (-not $customSteps) { $customSteps = "600" }

    $customVram = Read-Host "Enter VRAM preset (auto / 6g / 8g / 12g / 16g) [$defaultVramPreset]"
    if ($customVram) { $defaultVramPreset = $customVram }

    $customRes = Read-Host "Enter resolution (1024 / 768 / 512) [$defaultResolution]"
    if ($customRes) { $defaultResolution = $customRes }

    $customPrecision = Read-Host "Enter precision (bf16 / fp16) [bf16]"
    if (-not $customPrecision) { $customPrecision = "bf16" }

    Write-Host "`nStarting training with customized arguments..." -ForegroundColor Green
    & $pythonExe train_local.py `
        --data_dir "$defaultDataDir" `
        --output_dir "$outputDir" `
        --lora_name "$defaultLoraName" `
        --trigger_word "$defaultTriggerWord" `
        --train_mode "$defaultTrainMode" `
        --vram_preset "$defaultVramPreset" `
        --resolution $defaultResolution `
        --steps $customSteps `
        --precision $customPrecision `
        --copy_to_comfy_loras
} else {
    Write-Host "`nStarting standard production training..." -ForegroundColor Green
    & $pythonExe train_local.py `
        --data_dir "$defaultDataDir" `
        --output_dir "$outputDir" `
        --lora_name "$defaultLoraName" `
        --trigger_word "$defaultTriggerWord" `
        --train_mode "$defaultTrainMode" `
        --vram_preset "$defaultVramPreset" `
        --resolution $defaultResolution `
        --copy_to_comfy_loras
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n================================================================================" -ForegroundColor Green
    Write-Host ">>> Training Completed Successfully! Output saved to: $outputDir\$defaultLoraName <<<" -ForegroundColor Green
    Write-Host "================================================================================" -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] Training terminated with error code $LASTEXITCODE" -ForegroundColor Red
}
