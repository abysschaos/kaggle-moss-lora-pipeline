@echo off
cd /d "%~dp0"

:: Zero-console silent launch via Windows Script Host + pythonw.exe
if exist "%~dp0launch_ui.vbs" (
    start "" wscript.exe //nologo "%~dp0launch_ui.vbs"
    exit /b 0
)

:: Fallback: direct detached launch via silent_launcher.pyw
if exist "%~dp0venv\Scripts\pythonw.exe" (
    start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0scripts\silent_launcher.pyw"
    exit /b 0
)

start "" pythonw.exe "%~dp0scripts\silent_launcher.pyw"
exit /b 0
