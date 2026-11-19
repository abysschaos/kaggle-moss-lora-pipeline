using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Windows.Forms;
using Microsoft.Win32;

[assembly: AssemblyTitle("OpenMOSS MOSS-Image LoRA Trainer Launcher")]
[assembly: AssemblyDescription("Zero-console native Windows launcher for OpenMOSS MOSS-Image LoRA Trainer")]
[assembly: AssemblyProduct("OpenMOSS MOSS-Image LoRA Trainer")]
[assembly: AssemblyVersion("1.1.0.0")]
[assembly: AssemblyFileVersion("1.1.0.0")]

static class MossTrainerLauncher
{
    [STAThread]
    static void Main(string[] args)
    {
        try
        {
            string rootDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\', '/');
            string guiScript = Path.Combine(rootDir, "gui_local.py");
            string fallbackLauncher = Path.Combine(rootDir, "scripts", "silent_launcher.pyw");

            if (!File.Exists(guiScript))
            {
                MessageBox.Show(
                    "未找到主程序脚本：\n" + guiScript + "\n\n请将 MOSS_LoRA_Trainer.exe 保持在项目根目录内（如需桌面启动请右键创建快捷方式）。",
                    "OpenMOSS MOSS-Image LoRA 训练器",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            // 1. 极速直达通道：直接定位高优先级 PyTorch 环境（venv / COMFYUI_PYTHON / local_config.json / 常见便携路径）
            //    命中后一步直接启动 gui_local.py，免去二次拉起 Python 进程的开销。
            string directPy = ResolveDirectPyTorchPython(rootDir);
            if (!string.IsNullOrEmpty(directPy))
            {
                LaunchProcess(directPy, guiScript, rootDir);
                return;
            }

            // 2. 全盘深度探测兜底：使用系统任意可用 pythonw.exe / python.exe 拉起 scripts/silent_launcher.pyw，
            //    自动复用 train_local.py 的全盘宽域 ComfyUI 扫描能力。
            string sysPy = ResolveSystemPython();
            if (!string.IsNullOrEmpty(sysPy) && File.Exists(fallbackLauncher))
            {
                LaunchProcess(sysPy, fallbackLauncher, rootDir);
                return;
            }

            if (!string.IsNullOrEmpty(sysPy))
            {
                LaunchProcess(sysPy, guiScript, rootDir);
                return;
            }

            MessageBox.Show(
                "未检测到可用的 Python 运行环境。\n\n解决方案：\n1. 双击根目录 install_windows.bat 自动部署虚拟环境；\n2. 或安装 Python 3.10+ 并勾选 Add to PATH；\n3. 或设置环境变量 COMFYUI_PYTHON 指向您的 ComfyUI python.exe。",
                "OpenMOSS MOSS-Image LoRA 训练器",
                MessageBoxButtons.OK,
                MessageBoxIcon.Warning
            );
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                "启动控制台时发生异常：\n" + ex.Message,
                "OpenMOSS MOSS-Image LoRA 训练器",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
        }
    }

    static void LaunchProcess(string pythonExe, string scriptPath, string rootDir)
    {
        var psi = new ProcessStartInfo
        {
            FileName = pythonExe,
            Arguments = "\"" + scriptPath + "\"",
            WorkingDirectory = rootDir,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        string pyDir = Path.GetDirectoryName(pythonExe) ?? "";
        string scriptsDir = Path.Combine(pyDir, "Scripts");
        string existingPath = Environment.GetEnvironmentVariable("PATH") ?? "";
        psi.EnvironmentVariables["PATH"] = scriptsDir + ";" + pyDir + ";" + existingPath;

        Process.Start(psi);
    }

    static string PreferPythonw(string pyPath)
    {
        if (string.IsNullOrWhiteSpace(pyPath)) return null;
        try
        {
            string dir = Path.GetDirectoryName(pyPath);
            if (!string.IsNullOrEmpty(dir))
            {
                string pyw = Path.Combine(dir, "pythonw.exe");
                if (File.Exists(pyw)) return pyw;
            }
            if (File.Exists(pyPath)) return pyPath;
        }
        catch { }
        return null;
    }

    static bool HasTorchPackage(string pythonExe)
    {
        try
        {
            string dir = Path.GetDirectoryName(pythonExe);
            if (string.IsNullOrEmpty(dir)) return false;
            if (Directory.Exists(Path.Combine(dir, "Lib", "site-packages", "torch"))) return true;
            // venv 情况下 pythonw.exe 位于 venv\Scripts\，site-packages 位于 venv\Lib\site-packages
            string parent = Path.GetDirectoryName(dir);
            if (!string.IsNullOrEmpty(parent) && Directory.Exists(Path.Combine(parent, "Lib", "site-packages", "torch")))
            {
                return true;
            }
        }
        catch { }
        return false;
    }

    static string ResolveDirectPyTorchPython(string rootDir)
    {
        // (1) 项目独立虚拟环境 venv
        string venvPy = PreferPythonw(Path.Combine(rootDir, "venv", "Scripts", "python.exe"));
        if (!string.IsNullOrEmpty(venvPy)) return venvPy;

        // (2) 显式环境变量 COMFYUI_PYTHON
        string envPy = Environment.GetEnvironmentVariable("COMFYUI_PYTHON");
        if (!string.IsNullOrWhiteSpace(envPy))
        {
            string resolved = PreferPythonw(envPy.Trim().Trim('"'));
            if (!string.IsNullOrEmpty(resolved)) return resolved;
        }

        // (3) 从本地 local_config.json 读取上次记住的 comfy_models_dir 并推导同级 ComfyUI Python
        string savedModelsDir = ReadJsonStringField(Path.Combine(rootDir, "local_config.json"), "comfy_models_dir");
        if (!string.IsNullOrWhiteSpace(savedModelsDir) && Directory.Exists(savedModelsDir))
        {
            var probeRoots = new List<string>();
            try
            {
                DirectoryInfo cur = new DirectoryInfo(savedModelsDir);
                for (int depth = 0; depth < 3 && cur != null; depth++)
                {
                    probeRoots.Add(cur.FullName);
                    cur = cur.Parent;
                }
            }
            catch { }

            string[] relCandidates = new[]
            {
                Path.Combine("python", "python.exe"),
                Path.Combine("python_embeded", "python.exe"),
                Path.Combine("python_embedded", "python.exe"),
                Path.Combine("venv", "Scripts", "python.exe")
            };

            foreach (string baseDir in probeRoots)
            {
                foreach (string rel in relCandidates)
                {
                    string cand = PreferPythonw(Path.Combine(baseDir, rel));
                    if (!string.IsNullOrEmpty(cand) && HasTorchPackage(cand))
                    {
                        return cand;
                    }
                }
            }
        }

        // (4) 常见盘符便携版快速直达
        foreach (string drv in new[] { "C:", "D:", "E:", "F:", "G:" })
        {
            foreach (string rel in new[]
            {
                @"ComfyUI_windows_portable\python_embeded\python.exe",
                @"ComfyUI\python\python.exe"
            })
            {
                string cand = PreferPythonw(drv + "\\" + rel);
                if (!string.IsNullOrEmpty(cand) && HasTorchPackage(cand))
                {
                    return cand;
                }
            }
        }

        return null;
    }

    static string ResolveSystemPython()
    {
        // (1) PATH 环境变量检索
        string pathEnv = Environment.GetEnvironmentVariable("PATH") ?? "";
        foreach (string exeName in new[] { "pythonw.exe", "python.exe" })
        {
            foreach (string dir in pathEnv.Split(';'))
            {
                if (string.IsNullOrWhiteSpace(dir)) continue;
                try
                {
                    string cleanDir = dir.Trim().Trim('"');
                    // 跳过 Microsoft Store 的零字节空壳占位符
                    if (cleanDir.IndexOf("WindowsApps", StringComparison.OrdinalIgnoreCase) >= 0) continue;
                    string full = Path.Combine(cleanDir, exeName);
                    if (File.Exists(full)) return PreferPythonw(full);
                }
                catch { }
            }
        }

        // (2) Windows 注册表 PythonCore 安装路径回退
        RegistryKey[] roots = new[] { Registry.CurrentUser, Registry.LocalMachine };
        foreach (RegistryKey hive in roots)
        {
            try
            {
                using (RegistryKey pyCore = hive.OpenSubKey(@"SOFTWARE\Python\PythonCore"))
                {
                    if (pyCore == null) continue;
                    string[] versions = pyCore.GetSubKeyNames();
                    Array.Sort(versions, StringComparer.OrdinalIgnoreCase);
                    Array.Reverse(versions);
                    foreach (string ver in versions)
                    {
                        using (RegistryKey ip = pyCore.OpenSubKey(ver + @"\InstallPath"))
                        {
                            if (ip == null) continue;
                            string exePath = ip.GetValue("ExecutablePath") as string;
                            string resolved = PreferPythonw(exePath);
                            if (!string.IsNullOrEmpty(resolved)) return resolved;

                            string installDir = ip.GetValue("") as string;
                            if (!string.IsNullOrEmpty(installDir))
                            {
                                resolved = PreferPythonw(Path.Combine(installDir, "python.exe"));
                                if (!string.IsNullOrEmpty(resolved)) return resolved;
                            }
                        }
                    }
                }
            }
            catch { }
        }

        return null;
    }

    static string ReadJsonStringField(string jsonPath, string fieldName)
    {
        try
        {
            if (!File.Exists(jsonPath)) return null;
            string text = File.ReadAllText(jsonPath, System.Text.Encoding.UTF8);
            string pattern = "\"" + Regex.Escape(fieldName) + "\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"";
            Match m = Regex.Match(text, pattern);
            if (!m.Success) return null;
            return Regex.Unescape(m.Groups[1].Value);
        }
        catch
        {
            return null;
        }
    }
}
