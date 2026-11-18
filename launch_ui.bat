@echo off
chcp 65001 >nul
title OpenMOSS Local LoRA Trainer GUI
cd /d "%~dp0"
echo Starting OpenMOSS MOSS-Image Local Trainer GUI...
python gui_local.py
if errorlevel 1 (
    echo.
    echo [ERROR] GUI launch failed! Press any key to exit.
    pause >nul
)
