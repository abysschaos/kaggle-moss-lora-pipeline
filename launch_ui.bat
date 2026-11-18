@echo off
cd /d "%~dp0"
title OpenMOSS Local LoRA Trainer GUI

:: 1. Check local project virtual environment (venv)
if exist "%~dp0venv\Scripts\python.exe" goto use_venv

:: 2. Check user-defined custom environment variable
if defined COMFYUI_PYTHON if exist "%COMFYUI_PYTHON%" set "TARGET_PY=%COMFYUI_PYTHON%"
if defined TARGET_PY goto use_target

:: 3. Dynamic broad-range auto discovery across all drives and directories
for /f "usebackq delims=" %%I in (`python "%~dp0scripts\detect_env.py" 2^>nul`) do if exist "%%~fI" set "TARGET_PY=%%~fI"
if defined TARGET_PY goto use_target

:: 4. Fast check for standard portable paths across typical drives
if exist "%~dp0..\ComfyUI_windows_portable\python_embeded\python.exe" set "TARGET_PY=%~dp0..\ComfyUI_windows_portable\python_embeded\python.exe"
if defined TARGET_PY goto use_target

for %%D in (C: D: E: F: G:) do if exist "%%D\ComfyUI_windows_portable\python_embeded\python.exe" set "TARGET_PY=%%D\ComfyUI_windows_portable\python_embeded\python.exe"
if defined TARGET_PY goto use_target

for %%D in (C: D: E: F: G:) do if exist "%%D\ComfyUI\python\python.exe" set "TARGET_PY=%%D\ComfyUI\python\python.exe"
if defined TARGET_PY goto use_target

:: 5. Fallback to system Python
echo [INFO] Falling back to system Python...
python "%~dp0gui_local.py"
goto finish

:use_venv
echo [INFO] Launching with local venv: "%~dp0venv"
set "PATH=%~dp0venv\Scripts;%PATH%"
call "%~dp0venv\Scripts\activate.bat"
python "%~dp0gui_local.py"
goto finish

:use_target
echo [INFO] Detected high-performance PyTorch environment:
echo        "%TARGET_PY%"
"%TARGET_PY%" "%~dp0gui_local.py"
goto finish

:finish
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] GUI terminated with error code %errorlevel%.
    pause
)
