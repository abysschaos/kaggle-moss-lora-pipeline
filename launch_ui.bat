@echo off
cd /d "%~dp0"
title OpenMOSS Local LoRA Trainer GUI

if exist "venv\Scripts\python.exe" goto use_venv
goto use_system

:use_venv
echo [INFO] Launching GUI with local venv...
call "venv\Scripts\activate.bat"
python gui_local.py
goto end

:use_system
echo [INFO] Launching GUI with system Python...
python gui_local.py
goto end

:end
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] GUI terminated with error code %errorlevel%.
    pause
)
