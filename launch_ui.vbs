Option Explicit

Dim fso, shell, scriptDir, launcherPyw, venvPyw, envPy, pyDir, candidatePyw, cmd
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
launcherPyw = fso.BuildPath(scriptDir, "scripts\silent_launcher.pyw")
venvPyw = fso.BuildPath(scriptDir, "venv\Scripts\pythonw.exe")

shell.CurrentDirectory = scriptDir

' Priority 1: Local project venv pythonw.exe
If fso.FileExists(venvPyw) Then
    cmd = """" & venvPyw & """ """ & launcherPyw & """"
    shell.Run cmd, 0, False
    WScript.Quit 0
End If

' Priority 2: User-defined COMFYUI_PYTHON sibling pythonw.exe
envPy = shell.ExpandEnvironmentStrings("%COMFYUI_PYTHON%")
If envPy <> "" And envPy <> "%COMFYUI_PYTHON%" And fso.FileExists(envPy) Then
    pyDir = fso.GetParentFolderName(envPy)
    candidatePyw = fso.BuildPath(pyDir, "pythonw.exe")
    If fso.FileExists(candidatePyw) Then
        cmd = """" & candidatePyw & """ """ & launcherPyw & """"
    Else
        cmd = """" & envPy & """ """ & launcherPyw & """"
    End If
    shell.Run cmd, 0, False
    WScript.Quit 0
End If

' Priority 3: System pythonw.exe (0 = completely hidden, no CMD/PowerShell window)
On Error Resume Next
cmd = "pythonw.exe """ & launcherPyw & """"
shell.Run cmd, 0, False
If Err.Number <> 0 Then
    Err.Clear
    cmd = "python.exe """ & launcherPyw & """"
    shell.Run cmd, 0, False
End If
On Error GoTo 0
