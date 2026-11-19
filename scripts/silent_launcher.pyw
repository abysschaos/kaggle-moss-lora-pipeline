"""
OpenMOSS MOSS-Image 零黑框静默 GUI 启动引导器 (Windows 原生桌面体验)
自动寻址本地 venv / ComfyUI 高性能 PyTorch 环境，并以 pythonw + CREATE_NO_WINDOW 无黑框启动 gui_local.py。
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _prefer_pythonw(py_path: Path) -> Path:
    """优先返回同目录下的 pythonw.exe，确保启动 GUI 时不分配黑框控制台。"""
    pyw = py_path.with_name("pythonw.exe")
    return pyw if pyw.is_file() else py_path


def resolve_best_python() -> Path:
    # 1. 项目独立虚拟环境 (venv)
    venv_py = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    if venv_py.is_file():
        return _prefer_pythonw(venv_py)

    # 2. 用户自定义环境变量 COMFYUI_PYTHON
    env_py = os.environ.get("COMFYUI_PYTHON", "").strip()
    if env_py and Path(env_py).is_file():
        return _prefer_pythonw(Path(env_py))

    # 3. 宽域自动探测 ComfyUI 便携版 / 整合包 Python
    try:
        import train_local
        _, pythons = train_local.find_broad_comfy_paths()
        for py in pythons:
            if py.is_file():
                return _prefer_pythonw(py)
    except Exception:
        pass

    # 4. 常见盘符快速回退扫描
    for drv in ("C:", "D:", "E:", "F:", "G:"):
        for rel in (
            "ComfyUI_windows_portable/python_embeded/python.exe",
            "ComfyUI/python/python.exe",
        ):
            cand = Path(f"{drv}/{rel}")
            if cand.is_file():
                return _prefer_pythonw(cand)

    # 5. 回退至当前解释器
    return _prefer_pythonw(Path(sys.executable))


def main():
    target_py = resolve_best_python()
    gui_script = PROJECT_ROOT / "gui_local.py"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    # 同步把目标 Python 的 Scripts 目录注入 PATH，保障内部依赖与可执行工具寻址无阻
    env = os.environ.copy()
    py_dir = str(target_py.parent)
    scripts_dir = str(target_py.parent / "Scripts")
    env["PATH"] = f"{scripts_dir};{py_dir};{env.get('PATH', '')}"

    subprocess.Popen(
        [str(target_py), str(gui_script)],
        cwd=str(PROJECT_ROOT),
        env=env,
        creationflags=creation_flags,
        close_fds=True,
    )


if __name__ == "__main__":
    main()
