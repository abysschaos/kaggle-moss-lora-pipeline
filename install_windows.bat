@echo off
:: ==============================================================================
:: OpenMOSS MOSS-Image Windows 本地环境一键自动化安装脚本
:: 适用环境: Windows 10/11 64-bit | NVIDIA GPU (RTX 30/40系列推荐)
:: ==============================================================================

chcp 65001 >nul
title OpenMOSS MOSS-Image 一键安装程序
cd /d "%~dp0"
cls

echo ================================================================================
echo   OpenMOSS MOSS-Image 本地 LoRA 训练环境一键安装程序 (Windows 11/10)
echo ================================================================================
echo.

:: 1. 检测 Python
echo [1/6] 检查系统 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python 命令！
    echo 请先安装 Python 3.10 至 3.12 (推荐 3.12 64-bit)。
    echo 下载地址: https://www.python.org/downloads/
    echo 提示: 安装时请务必勾选 "Add python.exe to PATH"！
    echo.
    pause
    exit /b 1
)

:: 检查 Python 架构是否为 64-bit
python -c "import sys; exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
if errorlevel 1 (
    echo [错误] 当前 Python 是 32 位的，PyTorch CUDA 仅支持 64 位 Python！
    echo 请卸载并重新安装 64 位的 Python。
    echo.
    pause
    exit /b 1
)

python -c "import sys; print('   -> 检测到 Python 版本:', sys.version.split()[0], '(64-bit)')"

:: 2. 检测 Git
echo.
echo [2/6] 检查系统 Git 环境...
git --version >nul 2>&1
if errorlevel 1 (
    echo [警告] 未检测到 Git 命令，后续可能无法自动同步或克隆组件！
    echo 推荐安装 Git for Windows: https://git-scm.com/download/win
    echo 若已安装请确认已加入系统 PATH 环境变量。
    echo.
) else (
    git --version
)

:: 3. 创建专属隔离虚拟环境 venv
echo.
echo [3/6] 配置独立虚拟运行环境 (venv)...
if not exist "venv\" (
    echo    -> 正在创建虚拟环境 venv (耗时约 5~10 秒)...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败，请检查目录权限或 Python 完整性。
        pause
        exit /b 1
    )
    echo    -> 虚拟环境 venv 创建成功！
) else (
    echo    -> 检测到已存在的 venv 目录，跳过创建。
)

:: 激活虚拟环境
call "venv\Scripts\activate.bat"

:: 升级基础打包工具
echo    -> 升级 pip, setuptools 与 wheel...
python -m pip install --upgrade pip setuptools wheel -q

:: 4. 安装带 CUDA 加速的 PyTorch
echo.
echo [4/6] 安装 PyTorch 与 TorchVision (CUDA 12.4 驱动版本)...
echo    -> 正在拉取官方 CUDA 预编译轮子 (若网络缓慢请检查代理)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo.
    echo [重试] cu124 源拉取失败，尝试降级为 cu121 源...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo [错误] PyTorch 安装失败，请检查网络连接或手动安装。
        pause
        exit /b 1
    )
)

:: 5. 安装本地专用训练与生态辅助依赖
echo.
echo [5/6] 安装本地训练辅助依赖 (requirements-local.txt)...
pip install -r requirements-local.txt
if errorlevel 1 (
    echo [错误] 安装辅助依赖失败，请查看上方报错日志。
    pause
    exit /b 1
)

:: 6. 自动预创本地标准资产目录与说明
echo.
echo [6/6] 初始化本地资产目录结构...
if not exist "models\diffusion_models\" mkdir "models\diffusion_models"
if not exist "models\text_encoders\" mkdir "models\text_encoders"
if not exist "models\vae\" mkdir "models\vae"
if not exist "dataset\" mkdir "dataset"
if not exist "output\" mkdir "output"

:: 写入底模放置说明
> "models\diffusion_models\放置底模在此.txt" (
    echo [MOSS-Image 底模放置说明]
    echo 请将 MOSS-Image 底模权重放置于本目录。
    echo 推荐修复版: moss_image_heal_bf16.safetensors 或 moss_image_heal_fp16.safetensors
    echo 官方下载地址: https://huggingface.co/KBlueLeaf/MOSS-Image-heal/tree/main/comfyui
    echo.
    echo 提示: 若本地已有 ComfyUI 模型库，可无需下载，直接在 GUI 界面选择 ComfyUI 路径即可！
)

:: 写入文本编码器说明
> "models\text_encoders\放置文本编码器在此.txt" (
    echo [Qwen3.5-9B 文本编码器放置说明]
    echo 请将 Qwen3.5-9B 权重放置于本目录。
    echo 推荐文件: qwen3.5_9b_bf16.safetensors 或 qwen3.5_9b_int8_convrot.safetensors
    echo 官方下载地址: https://huggingface.co/Comfy-Org/MOSS-Image/tree/main/text_encoders
)

:: 写入 VAE 说明
> "models\vae\放置VAE在此.txt" (
    echo [FLUX.2 VAE 放置说明]
    echo 请将 FLUX.2 VAE 权重放置于本目录。
    echo 推荐文件: flux2-vae_small.safetensors (~168MB)
    echo 官方下载地址: https://huggingface.co/Comfy-Org/MOSS-Image/tree/main/vae
)

:: 写入训练集说明
> "dataset\数据排放说明.txt" (
    echo [训练集放置说明]
    echo 请在此目录下建立图集文件夹，或直接在任意路径选择您的训练文件夹。
    echo 文件夹内必须包含成对的 .jpg / .png 图片与同名 .txt 自然语言标注文件。
)

echo    -> 目录初始化完成！

:: 7. 环境最终可用性自检
echo.
echo ================================================================================
echo   正在执行硬件与 CUDA 最终自检...
echo ================================================================================
python -c "import torch; print('   PyTorch 版本:', torch.__version__); print('   CUDA 可用状态:', torch.cuda.is_available()); print('   显卡型号:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'); print('   BF16 硬件加速支持:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"

echo.
echo ================================================================================
echo   恭喜！本地部署环境已全部就绪！
echo ================================================================================
echo 下一步操作指引:
echo   1. 如果您本地已有 ComfyUI，无需重复下载底模，直接启动 GUI 选择路径即可；
echo   2. 若为纯净全新设备，请按照 models/ 对应文件夹内的 txt 指引放置底模；
echo   3. 双击运行根目录下的 launch_ui.bat 即可拉起图形化控制面板进行训练。
echo.
pause
