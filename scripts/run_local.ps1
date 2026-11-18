# ==============================================================================
# OpenMOSS MOSS-Image Windows Local Training Launcher
# Target Hardware: NVIDIA GeForce RTX 4070 Laptop / 4060 / 3060 (8GB - 12GB VRAM)
# ==============================================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "OpenMOSS MOSS-Image Local LoRA Trainer"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ">>> OpenMOSS MOSS-Image Local LoRA Training Launcher (Windows 11) <<<" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. 检查 Python 环境
$pythonCmd = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python is not installed or not in PATH! Please install Python 3.10 - 3.12." -ForegroundColor Red
    exit 1
}

Write-Host "[1/4] Verifying Python and PyTorch Environment..." -ForegroundColor Yellow
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

python -c $diagScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] PyTorch CUDA verification failed. Make sure PyTorch with CUDA is installed." -ForegroundColor Red
    exit 1
}

# 2. 检查默认路径配置（优先使用项目标准相对路径，杜绝任何个人隐私路径）
$defaultDataDir = ".\dataset"
if (-not (Test-Path $defaultDataDir) -and (Test-Path "$env:USERPROFILE\Pictures")) {
    $defaultDataDir = "$env:USERPROFILE\Pictures"
}
$defaultComfyModels = ".\models"
$outputDir = ".\output"

Write-Host "`n[2/4] Checking Local Dataset & Model Assets..." -ForegroundColor Yellow
if (Test-Path $defaultDataDir) {
    $imgCount = (Get-ChildItem $defaultDataDir -Filter *.jpg).Count + (Get-ChildItem $defaultDataDir -Filter *.png).Count
    $txtCount = (Get-ChildItem $defaultDataDir -Filter *.txt).Count
    Write-Host " [OK] Found Dataset at: $defaultDataDir ($imgCount images, $txtCount txt captions)" -ForegroundColor Green
} else {
    Write-Host " [WARNING] Default dataset folder not found at: $defaultDataDir" -ForegroundColor Yellow
}

if (Test-Path $defaultComfyModels) {
    Write-Host " [OK] Found ComfyUI Models directory at: $defaultComfyModels" -ForegroundColor Green
} else {
    Write-Host " [WARNING] Default ComfyUI models folder not found at: $defaultComfyModels" -ForegroundColor Yellow
}

# 3. 运行参数确认
$defaultLoraName = "my_lora"
$defaultTriggerWord = ""

Write-Host "`n[3/4] Default Training Profile for RTX 4070 (8GB VRAM):" -ForegroundColor Cyan
Write-Host " - LoRA Name:      $defaultLoraName (output subfolder and weights prefix)"
Write-Host " - Trigger Word:   $($defaultTriggerWord ? $defaultTriggerWord : '(None, using original captions)')"
Write-Host " - Precision:      BF16 (Ada Lovelace Tensor Core acceleration)"
Write-Host " - Text Device:    CPU (prevents Qwen3.5-9B OOM on 8GB VRAM)"
Write-Host " - VAE Device:     CUDA (flux2-vae_small, ~168MB)"
Write-Host " - Steps / Epochs: 600 Steps / 6 Epochs (Repeats=10)"
Write-Host " - LoRA Dim/Alpha: Rank=32, Alpha=32.0 (Full 1.0 Strength)"
Write-Host " - Learning Rate:  1e-4"
Write-Host " - Output Path:    $outputDir\$defaultLoraName"

$confirm = Read-Host "`nPress Enter to start training immediately (or type 'custom' to adjust arguments)"
if ($confirm -eq "custom") {
    $customName = Read-Host "Enter custom LoRA name / folder name [$defaultLoraName]"
    if ($customName) { $defaultLoraName = $customName }

    $customTrigger = Read-Host "Enter custom trigger word (e.g. makima_char) [None]"
    if ($customTrigger) { $defaultTriggerWord = $customTrigger }

    $customData = Read-Host "Enter dataset path [$defaultDataDir]"
    if ($customData) { $defaultDataDir = $customData }
    
    $customSteps = Read-Host "Enter training steps [600]"
    if (-not $customSteps) { $customSteps = "600" }

    $customPrecision = Read-Host "Enter precision (bf16/fp16) [bf16]"
    if (-not $customPrecision) { $customPrecision = "bf16" }

    Write-Host "`nStarting training with customized arguments..." -ForegroundColor Green
    python train_local.py `
        --data_dir "$defaultDataDir" `
        --comfy_models_dir "$defaultComfyModels" `
        --output_dir "$outputDir" `
        --lora_name "$defaultLoraName" `
        --trigger_word "$defaultTriggerWord" `
        --steps $customSteps `
        --precision $customPrecision `
        --copy_to_comfy_loras
} else {
    Write-Host "`nStarting standard production training..." -ForegroundColor Green
    python train_local.py `
        --data_dir "$defaultDataDir" `
        --comfy_models_dir "$defaultComfyModels" `
        --output_dir "$outputDir" `
        --lora_name "$defaultLoraName" `
        --trigger_word "$defaultTriggerWord" `
        --copy_to_comfy_loras
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n================================================================================" -ForegroundColor Green
    Write-Host ">>> Training Completed Successfully! Output saved to: $outputDir <<<" -ForegroundColor Green
    Write-Host "================================================================================" -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] Training terminated with error code $LASTEXITCODE" -ForegroundColor Red
}
