"""
OpenMOSS MOSS-Image 零黑框静默 GUI 启动引导器 (Windows 原生桌面体验)
自动寻址本地 venv / local_config ComfyUI / 全盘便携包高性能 PyTorch 环境，
严格校验目标 Python 已内置 torch 后，以 pythonw + CREATE_NO_WINDOW 无黑框启动 gui_local.py。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _prefer_pythonw(py_path: Path) -> Path:
    """优先返回同目录下的 pythonw.exe，确保启动 GUI 时不分配黑框控制台。"""
    pyw = py_path.with_name("pythonw.exe")
    return pyw if pyw.is_file() else py_path


def _has_torch(py_path: Path) -> bool:
    """零子进程快速检测解释器是否自带 torch（支持 portable python/ 与 venv/Scripts/）。"""
    if not py_path.is_file():
        return False
    py_dir = py_path.parent
    return (
        (py_dir / "Lib" / "site-packages" / "torch").is_dir()
        or (py_dir.parent / "Lib" / "site-packages" / "torch").is_dir()
    )


def _resolve_from_local_config() -> Path | None:
    cfg_file = PROJECT_ROOT / "local_config.json"
    if not cfg_file.is_file():
        return None
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        saved_models = str(cfg.get("comfy_models_dir", "") or "").strip()
        if not saved_models or not os.path.exists(saved_models):
            return None
        cur = Path(saved_models).resolve()
        for _ in range(3):
            for rel in (
                "python/python.exe",
                "python_embeded/python.exe",
                "python_embedded/python.exe",
                "venv/Scripts/python.exe",
            ):
                cand = cur / rel
                if _has_torch(cand):
                    return _prefer_pythonw(cand)
            if cur.parent == cur:
                break
            cur = cur.parent
    except Exception:
        pass
    return None


def resolve_best_python() -> Path:
    # 1. 项目独立虚拟环境 (venv)
    venv_py = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    if _has_torch(venv_py):
        return _prefer_pythonw(venv_py)

    # 2. 用户自定义环境变量 COMFYUI_PYTHON
    env_py = os.environ.get("COMFYUI_PYTHON", "").strip()
    if env_py and _has_torch(Path(env_py)):
        return _prefer_pythonw(Path(env_py))

    # 3. 从 local_config.json 已知 ComfyUI 路径直接向上寻址（免全盘遍历）
    cfg_py = _resolve_from_local_config()
    if cfg_py is not None:
        return cfg_py

    # 4. 宽域自动探测 ComfyUI 便携版 / 整合包 Python
    try:
        import train_local

        _, pythons = train_local.find_broad_comfy_paths()
        for py in pythons:
            if _has_torch(py):
                return _prefer_pythonw(py)
    except Exception:
        pass

    # 5. 常见盘符快速回退扫描
    for drv in ("C:", "D:", "E:", "F:", "G:"):
        for rel in (
            "ComfyUI_windows_portable/python_embeded/python.exe",
            "ComfyUI/python/python.exe",
        ):
            cand = Path(f"{drv}/{rel}")
            if _has_torch(cand):
                return _prefer_pythonw(cand)

    # 6. 回退至当前解释器
    return _prefer_pythonw(Path(sys.executable))


def main():
    target_py = resolve_best_python()
    gui_script = PROJECT_ROOT / "gui_local.py"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

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
