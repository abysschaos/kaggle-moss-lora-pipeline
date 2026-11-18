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

# 5. 校验环境
Write-Host "`nVerifying PyTorch CUDA..." -ForegroundColor Yellow
& $venvPython -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

Write-Host "`n[SUCCESS] Local environment installation finished!" -ForegroundColor Green
Write-Host "Double click launch_ui.bat to start the training GUI panel." -ForegroundColor Cyan
