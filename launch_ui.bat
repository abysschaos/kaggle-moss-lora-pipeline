@echo off
chcp 65001 >nul
cd /d "%~dp0"
title OpenMOSS Local LoRA Trainer GUI

echo ================================================================================
echo   OpenMOSS MOSS-Image 本地 LoRA 训练引擎启动引导
echo ================================================================================

:: 1. 优先检测项目专属虚拟运行环境
if exist "venv\Scripts\python.exe" (
    echo [环境选择] 正在调用项目专属虚拟环境 (venv)...
    set "PATH=%CD%\venv\Scripts;%PATH%"
    call "venv\Scripts\activate.bat"
    python gui_local.py
    goto end
)

:: 2. 智能探测本地已安装的秋叶 ComfyUI / 独立高配 PyTorch 环境
set "CANDIDATE_AKI=C:\dowmload idm\ComfyUI-aki-v3.2(1)\ComfyUI-aki-v3.2\python\python.exe"
if exist "%CANDIDATE_AKI%" (
    echo [环境选择] 自动命中本地秋叶 ComfyUI 高性能 PyTorch 环境:
    echo            %CANDIDATE_AKI%
    set "PATH=C:\dowmload idm\ComfyUI-aki-v3.2(1)\ComfyUI-aki-v3.2\python\Scripts;%PATH%"
    "%CANDIDATE_AKI%" gui_local.py
    goto end
)

:: 探测其他常见相对路径下的 ComfyUI 便携版环境
if exist "..\ComfyUI-aki-v3.2(1)\ComfyUI-aki-v3.2\python\python.exe" (
    echo [环境选择] 命中相对路径秋叶 ComfyUI Python 环境...
    "..\ComfyUI-aki-v3.2(1)\ComfyUI-aki-v3.2\python\python.exe" gui_local.py
    goto end
)
if exist "..\ComfyUI_windows_portable\python_embeded\python.exe" (
    echo [环境选择] 命中 ComfyUI 便携版嵌入式 Python 环境...
    "..\ComfyUI_windows_portable\python_embeded\python.exe" gui_local.py
    goto end
)
if exist "..\python\python.exe" (
    echo [环境选择] 命中同级嵌入式 Python 环境...
    "..\python\python.exe" gui_local.py
    goto end
)

:: 3. 回退尝试系统环境变量中的全局 Python
echo [环境选择] 未找到整合包专用环境，正在尝试调用系统全局 Python...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ------------------------------------------------------------------------------
    echo [状态提示] 当前系统全局 Python 尚未安装带 CUDA 的 PyTorch。
    echo            控制台将以界面浏览模式启动，右上角会显示 "PyTorch 未加载"。
    echo.
    echo 若需在本地直接开始 GPU 训练:
    echo   1. 可双击运行根目录下的 install_windows.bat 创建专属虚拟环境；
    echo   2. 或将您电脑已有的 PyTorch 环境 python.exe 路径配置于 launch_ui.bat 中。
    echo ------------------------------------------------------------------------------
    echo.
)

python gui_local.py
goto end

:end
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] GUI 退出异常，退出代码: %errorlevel%。
    pause
)
