<#
.SYNOPSIS
    一键透明编译 Windows 原生零黑框启动器 MOSS_LoRA_Trainer.exe（无需任何第三方编译器或 PyInstaller）。
.DESCRIPTION
    使用 Windows 系统原生自带的 .NET Framework 4.x 编译器 (csc.exe)，
    将明文源码 launcher/MossTrainerLauncher.cs 编译为约 10KB 的原生 GUI 可执行程序，
    并自动嵌入 assets/moss_trainer.ico 图标。
.PARAMETER CreateDesktopShortcut
    若指定此开关，编译完成后自动在桌面创建指向 MOSS_LoRA_Trainer.exe 的快捷方式。
#>
param(
    [switch]$CreateDesktopShortcut
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceFile  = Join-Path $ProjectRoot "launcher\MossTrainerLauncher.cs"
$IconFile    = Join-Path $ProjectRoot "assets\moss_trainer.ico"
$OutputExe   = Join-Path $ProjectRoot "MOSS_LoRA_Trainer.exe"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  OpenMOSS MOSS-Image 原生 Windows 启动器一键编译工具" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if (-not (Test-Path $SourceFile)) {
    Write-Host "[错误] 未找到启动器源码: $SourceFile" -ForegroundColor Red
    exit 1
}

# 自动查找 Windows 原生自带 .NET Framework 4.x C# 编译器
$CscCandidates = @(
    (Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
    (Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe")
)

$CscExe = $null
foreach ($cand in $CscCandidates) {
    if (Test-Path $cand) {
        $CscExe = $cand
        break
    }
}

if (-not $CscExe) {
    Write-Host "[错误] 未检测到系统内置 .NET Framework 4.x 编译器 (csc.exe)。" -ForegroundColor Red
    Write-Host "您仍可直接双击根目录 launch_ui.vbs 无黑框启动控制台。" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/2] 检测到系统原生编译器: $CscExe" -ForegroundColor Green

$CompileArgs = @(
    "/nologo",
    "/optimize+",
    "/utf8output",
    "/target:winexe",
    "/r:System.Windows.Forms.dll",
    "/out:$OutputExe"
)

if (Test-Path $IconFile) {
    $CompileArgs += "/win32icon:$IconFile"
    Write-Host "      已加载自定义图标: $IconFile" -ForegroundColor DarkCyan
} else {
    Write-Host "      未检测到图标文件，将使用默认应用图标" -ForegroundColor Yellow
}

$CompileArgs += $SourceFile

Write-Host "[2/2] 正在编译原生零黑框启动器..." -ForegroundColor Cyan
& $CscExe @CompileArgs
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $OutputExe)) {
    Write-Host "[错误] 编译失败，退出码: $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

$FileInfo = Get-Item $OutputExe
$SizeKB = [math]::Round($FileInfo.Length / 1024, 1)
Write-Host ""
Write-Host "[成功] 已生成原生启动器: $OutputExe ($SizeKB KB)" -ForegroundColor Green
Write-Host "       现在您可以直接双击 MOSS_LoRA_Trainer.exe 无黑框启动，或右键固定到任务栏。" -ForegroundColor Gray

if ($CreateDesktopShortcut) {
    try {
        $DesktopDir = [Environment]::GetFolderPath("Desktop")
        $ShortcutPath = Join-Path $DesktopDir "MOSS LoRA 训练器.lnk"
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
        $Shortcut.TargetPath = $OutputExe
        $Shortcut.WorkingDirectory = $ProjectRoot
        if (Test-Path $IconFile) {
            $Shortcut.IconLocation = "$IconFile,0"
        }
        $Shortcut.Description = "OpenMOSS MOSS-Image 本地 LoRA 训练引擎控制台"
        $Shortcut.Save()
        Write-Host "[附加] 已创建桌面快捷方式: $ShortcutPath" -ForegroundColor Green
    } catch {
        Write-Host "[提示] 桌面快捷方式创建失败: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}
