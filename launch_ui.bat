@echo off
cd /d "%~dp0"
title OpenMOSS Local LoRA Trainer GUI

set "AKI_PY=C:\dowmload idm\ComfyUI-aki-v3.2(1)\ComfyUI-aki-v3.2\python\python.exe"

if exist "%~dp0venv\Scripts\python.exe" goto use_venv
if exist "%AKI_PY%" goto use_aki
if exist "%~dp0..\ComfyUI_windows_portable\python_embeded\python.exe" goto use_portable
goto use_system

:use_venv
echo [INFO] Launching with local venv: "%~dp0venv"
set "PATH=%~dp0venv\Scripts;%PATH%"
call "%~dp0venv\Scripts\activate.bat"
python "%~dp0gui_local.py"
goto finish

:use_aki
echo [INFO] Launching with ComfyUI PyTorch environment:
echo        "%AKI_PY%"
set "PATH=C:\dowmload idm\ComfyUI-aki-v3.2(1)\ComfyUI-aki-v3.2\python\Scripts;%PATH%"
"%AKI_PY%" "%~dp0gui_local.py"
goto finish

:use_portable
echo [INFO] Launching with portable ComfyUI Python...
"%~dp0..\ComfyUI_windows_portable\python_embeded\python.exe" "%~dp0gui_local.py"
goto finish

:use_system
echo [INFO] Launching with system Python...
python "%~dp0gui_local.py"
goto finish

:finish
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] GUI terminated with error code %errorlevel%.
    pause
)
