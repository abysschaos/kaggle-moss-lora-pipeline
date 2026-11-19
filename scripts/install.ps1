# ==============================================================================
# OpenMOSS MOSS-Image Local Environment Installer (PowerShell Version)
# ==============================================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "OpenMOSS Local Trainer Environment Installer"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ">>> OpenMOSS MOSS-Image Local Environment Installer (PowerShell) <<<" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. 检查 Python 64-bit
$pythonCmd = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python command not found! Please install Python 3.10 - 3.12 (64-bit)." -ForegroundColor Red
    Write-Host "Download: https://www.python.org/downloads/"
    exit 1
}

$is64Bit = python -c "import sys; print(sys.maxsize > 2**32)"
if ($is64Bit -ne "True") {
    Write-Host "[ERROR] 32-bit Python detected. PyTorch CUDA requires 64-bit Python." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Python 64-bit verified." -ForegroundColor Green

# 2. 虚拟环境
$venvDir = Join-Path $projectRoot "venv"
if (-not (Test-Path $venvDir)) {
    Write-Host "`nCreating virtual environment in: $venvDir..." -ForegroundColor Yellow
    python -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

$venvPython = Join-Path $venvDir "Scripts\python.exe"
$venvPip = Join-Path $venvDir "Scripts\pip.exe"

# 3. 安装 PyTorch CUDA 12.4
Write-Host "`nInstalling PyTorch with CUDA 12.4..." -ForegroundColor Yellow
& $venvPip install --upgrade pip setuptools wheel -q
& $venvPip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nFallback to cu121..." -ForegroundColor Yellow
    & $venvPip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
}

# 4. 安装 requirements-local.txt
Write-Host "`nInstalling local auxiliary requirements..." -ForegroundColor Yellow
& $venvPip install -r (Join-Path $projectRoot "requirements-local.txt")

# 5. 初始化资产目录与训练器结构
Write-Host "`nInitializing project directories..." -ForegroundColor Yellow
$dirs = @("models\diffusion_models", "models\text_encoders", "models\vae", "dataset", "output", "KohakuMossTrainer")
foreach ($d in $dirs) {
    $fullPath = Join-Path $projectRoot $d
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath | Out-Null
    }
}
$trainerNotice = Join-Path $projectRoot "KohakuMossTrainer\训练器放置说明.txt"
if (-not (Test-Path $trainerNotice)) {
    @"
[KohakuMossTrainer 训练器框架放置说明]
官方源码: https://github.com/KohakuBlueleaf/KohakuMossTrainer

1. 推荐自动化方案: 启动 MOSS_LoRA_Trainer.exe（或 launch_ui.vbs），在【训练器框架管理】卡片点击【一键自动克隆/下载】。
2. 手动离线解压: 解压后必须确保本目录 (KohakuMossTrainer\) 下直接包含 pyproject.toml、src/ 等，严禁双层嵌套套娃结构！
3. 补丁注入: 放置后在 GUI 中点击【检查并注入补丁】即可。
"@ | Out-File -FilePath $trainerNotice -Encoding utf8
}

# 6. 校验环境
Write-Host "`nVerifying PyTorch CUDA..." -ForegroundColor Yellow
& $venvPython -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

Write-Host "`n[SUCCESS] Local environment installation finished!" -ForegroundColor Green
Write-Host "Recommended: run .\build_exe.bat once to compile MOSS_LoRA_Trainer.exe, or double-click launch_ui.vbs to start without console windows." -ForegroundColor Cyan
