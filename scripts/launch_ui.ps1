# ==============================================================================
# OpenMOSS MOSS-Image Local GUI Silent Launcher (PowerShell)
# ==============================================================================

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Get-Item (Join-Path $scriptDir "..")).FullName
$vbsLauncher = Join-Path $projectRoot "launch_ui.vbs"

if (Test-Path $vbsLauncher) {
    Start-Process -FilePath "wscript.exe" -ArgumentList "//nologo `"$vbsLauncher`"" -WindowStyle Hidden
    exit 0
}

$pywLauncher = Join-Path $projectRoot "scripts\silent_launcher.pyw"
$venvPyw = Join-Path $projectRoot "venv\Scripts\pythonw.exe"
if (Test-Path $venvPyw) {
    Start-Process -FilePath $venvPyw -ArgumentList "`"$pywLauncher`"" -WorkingDirectory $projectRoot -WindowStyle Hidden
} else {
    Start-Process -FilePath "pythonw.exe" -ArgumentList "`"$pywLauncher`"" -WorkingDirectory $projectRoot -WindowStyle Hidden
}
