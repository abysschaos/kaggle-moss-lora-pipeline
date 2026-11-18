@echo off
chcp 65001 >nul
title OpenMOSS Local LoRA Trainer GUI
cd /d "%~dp0"

:: 优先检测本地虚拟环境 venv
if exist "venv\Scripts\python.exe" (
    echo [启动中] 正在使用本地独立虚拟环境 (venv) 启动 GUI 控制面板...
    call "venv\Scripts\activate.bat"
    python gui_local.py
) else (
    echo [启动中] 正在使用系统全局 Python 启动 GUI 控制面板...
    python gui_local.py
)

if errorlevel 1 (
    echo.
    echo [错误] GUI 启动失败！若尚未配置运行环境，请先双击运行 install_windows.bat 进行初始化。
    echo 按任意键退出...
    pause >nul
)
