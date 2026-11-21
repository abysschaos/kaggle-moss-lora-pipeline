"""轻量级环境探测器：优先按 venv -> COMFYUI_PYTHON -> local_config -> 宽域扫描定位含 PyTorch 的 Python。"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from silent_launcher import resolve_best_python

    py = resolve_best_python()
    if py.name.lower() == "pythonw.exe":
        cli_py = py.with_name("python.exe")
        if cli_py.is_file():
            py = cli_py
    if py.is_file():
        print(str(py))
except Exception:
    pass
